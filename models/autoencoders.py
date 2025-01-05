import tensorflow as tf
import numpy as np

class BasicAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim=32):
        super(BasicAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self):
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.latent_dim, activation=None)
        ])
        return encoder
        
    def _build_decoder(self):
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(np.prod(self.input_shape), activation='sigmoid'),
            tf.keras.layers.Reshape(self.input_shape)
        ])
        return decoder
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim=32):
        super(ConvolutionalAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self):
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim)
        ])
        return encoder
        
    def _build_decoder(self):
        # Calculate the shape after flattening
        conv_shape = (self.input_shape[0] // 8, self.input_shape[1] // 8, 64)
        
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.Dense(conv_shape[0] * conv_shape[1] * conv_shape[2]),
            tf.keras.layers.Reshape(conv_shape),
            tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            tf.keras.layers.Conv2DTranspose(self.input_shape[-1], 3, activation='sigmoid', 
                                          strides=2, padding='same')
        ])
        return decoder
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim=32):
        super(VariationalAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Separate dense layers for mean and log variance
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(x)
        
        # Sampling layer
        z = Sampling()([z_mean, z_log_var])
        
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
    def _build_decoder(self):
        # Calculate the shape after flattening
        conv_shape = (self.input_shape[0] // 4, self.input_shape[1] // 4, 64)
        
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.Dense(conv_shape[0] * conv_shape[1] * conv_shape[2], activation='relu'),
            tf.keras.layers.Reshape(conv_shape),
            tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            tf.keras.layers.Conv2D(self.input_shape[-1], 3, activation='sigmoid', padding='same')
        ], name='decoder')
        return decoder
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # Add KL divergence regularization loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        self.add_loss(kl_loss)
        
        return reconstructed
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=[1, 2, 3]
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

class Sampling(tf.keras.layers.Layer):
    """Reparameterization trick by sampling from a Gaussian."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
