import tensorflow as tf
import numpy as np

class BasicAutoencoder:
    def __init__(self, input_shape, latent_dim=32):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.model = self._build_autoencoder()
        
    def _build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        latent = tf.keras.layers.Dense(self.latent_dim)(x)
        return tf.keras.Model(inputs, latent, name='encoder')
        
    def _build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(latent_inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(np.prod(self.input_shape), activation='sigmoid')(x)
        outputs = tf.keras.layers.Reshape(self.input_shape)(outputs)
        return tf.keras.Model(latent_inputs, outputs, name='decoder')
        
    def _build_autoencoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return tf.keras.Model(inputs, outputs, name='autoencoder')

class ConvolutionalAutoencoder(BasicAutoencoder):
    def _build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        latent = tf.keras.layers.Dense(self.latent_dim)(x)
        return tf.keras.Model(inputs, latent, name='conv_encoder')

    def _build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(7 * 7 * 64)(latent_inputs)
        x = tf.keras.layers.Reshape((7, 7, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(latent_inputs, outputs, name='conv_decoder')

class VariationalAutoencoder:
    def __init__(self, input_shape, latent_dim=32):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.model = self._build_vae()
        
    def _build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        
        mean = tf.keras.layers.Dense(self.latent_dim)(x)
        log_var = tf.keras.layers.Dense(self.latent_dim)(x)
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
        z = tf.keras.layers.Lambda(sampling)([mean, log_var])
        return tf.keras.Model(inputs, [mean, log_var, z], name='encoder')
        
    def _build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(7 * 7 * 64)(latent_inputs)
        x = tf.keras.layers.Reshape((7, 7, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(latent_inputs, outputs, name='decoder')
        
    def _build_vae(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        mean, log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        vae = tf.keras.Model(inputs, outputs, name='vae')
        
        # Add KL divergence regularization loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var - tf.square(mean) - tf.exp(log_var)
        )
        vae.add_loss(kl_loss)
        return vae
