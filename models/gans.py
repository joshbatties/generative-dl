import tensorflow as tf

class DCGAN:
    def __init__(self, latent_dim=100, image_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.model = self._build_gan()
        
    def _build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 128, input_dim=self.latent_dim),
            tf.keras.layers.Reshape((7, 7, 128)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(self.image_shape[-1], kernel_size=5, padding='same', activation='tanh')
        ], name='generator')
        return model
        
    def _build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same',
                                 input_shape=self.image_shape),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        return model
        
    def _build_gan(self):
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        self.discriminator.trainable = False
        
        gan_input = tf.keras.Input(shape=(self.latent_dim,))
        fake_img = self.generator(gan_input)
        gan_output = self.discriminator(fake_img)
        
        model = tf.keras.Model(gan_input, gan_output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
        return model
        
    def train(self, dataset, epochs, batch_size=128):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            for batch in dataset:
                # Train Discriminator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                generated_images = self.generator.predict(noise)
                
                d_loss_real = self.discriminator.train_on_batch(batch, real)
                d_loss_fake = self.discriminator.train_on_batch(generated_images, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Train Generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.model.train_on_batch(noise, real)
                
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

class WGAN(DCGAN):
    def _build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same',
                                 input_shape=self.image_shape),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)  # No activation for Wasserstein loss
        ], name='critic')
        return model
        
    def _wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)
        
    def _build_gan(self):
        self.discriminator.compile(
            loss=self._wasserstein_loss,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005),
            metrics=['accuracy']
        )
        self.discriminator.trainable = False
        
        gan_input = tf.keras.Input(shape=(self.latent_dim,))
        fake_img = self.generator(gan_input)
        gan_output = self.discriminator(fake_img)
        
        model = tf.keras.Model(gan_input, gan_output)
        model.compile(
            loss=self._wasserstein_loss,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        )
        return model
