import tensorflow as tf
import numpy as np

class DCGAN:
    def __init__(self, latent_dim=100, image_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        self.gan = self._build_gan()
    
    def _build_generator(self):
        # Calculate output shapes for transpose convolutions
        h0, w0 = self.image_shape[0] // 4, self.image_shape[1] // 4
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            
            # Dense layer to reshape
            tf.keras.layers.Dense(h0 * w0 * 256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Reshape((h0, w0, 256)),
            
            # Transpose convolutions
            tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            
            tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            
            tf.keras.layers.Conv2D(self.image_shape[-1], 4, padding='same', activation='tanh')
        ], name='generator')
        
        return model
    
    def _build_discriminator(self):
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.image_shape),
            
            # Convolutional layers
            tf.keras.layers.Conv2D(64, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Conv2D(256, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.3),
            
            # Output layer
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        
        return model
    
    def _build_gan(self):
        # Make discriminator non-trainable for combined model
        self.discriminator.trainable = False
        
        # GAN input (noise) will be latent_dim-dimensional vectors
        gan_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        
        # Generator output
        x = self.generator(gan_input)
        
        # Discriminator determines validity
        gan_output = self.discriminator(x)
        
        # Compile GAN
        model = tf.keras.Model(gan_input, gan_output)
        return model
    
    def compile(self, d_optimizer=None, g_optimizer=None):
        if d_optimizer is None:
            d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        if g_optimizer is None:
            g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=d_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Compile the combined model
        self.gan.compile(
            optimizer=g_optimizer,
            loss='binary_crossentropy'
        )
    
    @tf.function
    def train_step(self, real_images, batch_size):
        # Generate random noise
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        # Generate fake images
        generated_images = self.generator(noise, training=True)
        
        # Labels for real and fake images
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        
        # Add random noise to labels for smoothing
        real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
        fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))
        
        # Train discriminator
        with tf.GradientTape() as tape:
            # Predictions
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(generated_images, training=True)
            
            # Losses
            d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_predictions)
            d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_predictions)
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
        
        # Apply gradients
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        # Train generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        misleading_labels = tf.ones((batch_size, 1))
        
        with tf.GradientTape() as tape:
            # Generate images and get predictions
            generated_images = self.generator(noise, training=True)
            predictions = self.discriminator(generated_images, training=False)
            
            # Generator loss
            g_loss = tf.keras.losses.binary_crossentropy(misleading_labels, predictions)
            g_loss = tf.reduce_mean(g_loss)
        
        # Apply gradients
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.gan.optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        return {'d_loss': d_loss, 'g_loss': g_loss}
    
    def train(self, dataset, epochs, batch_size=32):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            losses = []
            
            for batch in dataset:
                loss = self.train_step(batch, batch_size)
                losses.append(loss)
            
            # Calculate average losses
            d_loss = np.mean([l['d_loss'] for l in losses])
            g_loss = np.mean([l['g_loss'] for l in losses])
            
            print(f"D loss: {d_loss:.4f}, G loss: {g_loss:.4f}")
    
    def generate(self, num_images):
        noise = tf.random.normal([num_images, self.latent_dim])
        generated_images = self.generator(noise, training=False)
        return generated_images


class WGAN(DCGAN):
    def _build_discriminator(self):
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.image_shape),
            
            # Convolutional layers
            tf.keras.layers.Conv2D(64, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            
            tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            
            tf.keras.layers.Conv2D(256, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            
            # Output layer (no activation)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ], name='critic')
        
        return model
    
    def compile(self, d_optimizer=None, g_optimizer=None):
        if d_optimizer is None:
            d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        if g_optimizer is None:
            g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        
        # Compile critic (discriminator)
        self.discriminator.compile(
            optimizer=d_optimizer,
            loss=self._wasserstein_loss
        )
        
        # Compile generator
        self.gan.compile(
            optimizer=g_optimizer,
            loss=self._wasserstein_loss
        )
    
    def _wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)
    
    @tf.function
    def _gradient_penalty(self, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images + alpha * (fake_images - real_images)
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, interpolated)[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    @tf.function
    def train_step(self, real_images, batch_size):
        n_critic = 5
        gp_weight = 10.0
        
        for _ in range(n_critic):
            # Generate random noise
            noise = tf.random.normal([batch_size, self.latent_dim])
            
            # Generate fake images
            with tf.GradientTape() as tape:
                fake_images = self.generator(noise, training=True)
                
                # Get critic predictions
                real_pred = self.discriminator(real_images, training=True)
                fake_pred = self.discriminator(fake_images, training=True)
                
                # Wasserstein loss
                critic_loss = tf.reduce_mean(fake_pred - real_pred)
                
                # Gradient penalty
                gp = self._gradient_penalty(real_images, fake_images)
                critic_loss += gp_weight * gp
            
            # Apply gradients
            critic_grads = tape.gradient(
                critic_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(critic_grads, self.discriminator.trainable_variables))
        
        # Train generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_images, training=True)
            gen_loss = -tf.reduce_mean(fake_pred)
        
        # Apply gradients
        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gan.optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables))
        
        return {'d_loss': critic_loss, 'g_loss': gen_loss}
