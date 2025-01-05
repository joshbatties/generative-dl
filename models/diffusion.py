import tensorflow as tf
import numpy as np

class TimestepEmbedding(tf.keras.layers.Layer):
    """Embeds scalar timesteps into vectors, as described in DDPM paper."""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def call(self, timesteps):
        half = self.dim // 2
        freqs = tf.exp(
            -tf.math.log(self.max_period) * tf.range(half, dtype=tf.float32) / half
        )
        args = tf.cast(timesteps, dtype=tf.float32)[:, None] * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if self.dim % 2:
            embedding = tf.pad(embedding, [[0, 0], [0, 1]])
        return embedding

def get_diffusion_schedule(num_diffusion_steps):
    """Returns cosine variance schedule as in improved DDPM paper."""
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = tf.range(steps, dtype=tf.float32) / steps
        alphas_cumprod = tf.cos((x + s) / (1 + s) * tf.constant(np.pi) * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return tf.clip_by_value(betas, 0, 0.999)

    betas = cosine_beta_schedule(num_diffusion_steps)
    alphas = 1. - betas
    alphas_cumprod = tf.math.cumprod(alphas, axis=0)
    alphas_cumprod_prev = tf.pad(alphas_cumprod[:-1], [[1, 0]], constant_values=1.0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
    }

class DiffusionModel(tf.keras.Model):
    """Implementation of DDPM (Ho et al., 2020) & improved DDPM."""
    
    def __init__(self, 
                 timesteps=1000,
                 img_size=32,
                 img_channels=3,
                 embedding_dims=256,
                 widths=(128, 256, 256, 512),
                 block_depth=2,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.timesteps = timesteps
        self.img_size = img_size
        self.img_channels = img_channels
        
        # Diffusion schedule
        self.diffusion_schedule = get_diffusion_schedule(timesteps)
        
        # Network components
        self.timestep_embed = TimestepEmbedding(embedding_dims)
        self.downblocks = []
        self.upblocks = []
        
        # Build U-Net architecture
        input_channels = img_channels
        for width in widths:
            for _ in range(block_depth):
                self.downblocks.append(
                    self._build_resblock(width, embedding_dims))
            if width != widths[-1]:
                self.downblocks.append(self._build_downsample(width))
                input_channels = width
        
        self.bottleneck = [
            self._build_resblock(widths[-1], embedding_dims),
            self._build_resblock(widths[-1], embedding_dims),
        ]
        
        for width in reversed(widths):
            for _ in range(block_depth):
                self.upblocks.append(
                    self._build_resblock(width, embedding_dims))
            if width != widths[0]:
                self.upblocks.append(self._build_upsample(width))
        
        self.final_conv = tf.keras.layers.Conv2D(img_channels, kernel_size=3, padding='same')
        
    def _build_resblock(self, channels, emb_dim):
        def apply(x, emb, skips=None):
            inputs = x
            x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)
            x = tf.keras.layers.Activation('swish')(x)
            
            # Add timestep embedding
            emb = tf.keras.layers.Dense(channels)(emb)
            emb = tf.reshape(emb, [-1, 1, 1, channels])
            x = x + emb
            
            x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)
            
            if skips is not None:
                x = tf.concat([x, skips], axis=-1)
            
            return x + inputs
        return apply
    
    def _build_downsample(self, channels):
        def apply(x, emb):
            return tf.keras.layers.Conv2D(
                channels, 4, strides=2, padding='same')(x)
        return apply
    
    def _build_upsample(self, channels):
        def apply(x, emb):
            return tf.keras.layers.Conv2DTranspose(
                channels, 4, strides=2, padding='same')(x)
        return apply
    
    def diffuse(self, x_0, noise=None, t=None):
        """Adds noise to images according to diffusion schedule."""
        batch_size = tf.shape(x_0)[0]
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_0))
        if t is None:
            t = tf.random.uniform(
                shape=(batch_size,),
                minval=0,
                maxval=self.timesteps,
                dtype=tf.int32
            )
            
        alpha_cumprod = tf.gather(self.diffusion_schedule['alphas_cumprod'], t)
        alpha_cumprod = tf.reshape(alpha_cumprod, (-1, 1, 1, 1))
        
        # Combine signal and noise according to diffusion schedule
        x_t = (
            tf.sqrt(alpha_cumprod) * x_0 +
            tf.sqrt(1 - alpha_cumprod) * noise
        )
        return x_t, noise, t
    
    def call(self, inputs, training=False):
        """Predicts noise from noisy images and timesteps."""
        x, t = inputs
        
        # Timestep embedding
        temb = self.timestep_embed(t)
        temb = tf.keras.layers.Dense(temb.shape[-1] * 2, activation='swish')(temb)
        temb = tf.keras.layers.Dense(temb.shape[-1])(temb)
        
        # Downsampling
        skips = []
        x = tf.cast(x, self.compute_dtype)
        for block in self.downblocks:
            if 'Conv2D' in block.__class__.__name__:
                x = block(x)
                skips.append(x)
            else:
                x = block(x, temb)
                skips.append(x)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x, temb)
        
        # Upsampling
        for block in self.upblocks:
            if 'Conv2DTranspose' in block.__class__.__name__:
                x = block(x)
            else:
                if skips:
                    x = tf.concat([x, skips.pop()], axis=-1)
                x = block(x, temb)
        
        # Final prediction
        x = self.final_conv(x)
        return x
    
    def train_step(self, images):
        """Custom train step to implement diffusion training."""
        with tf.GradientTape() as tape:
            # Add noise to images
            x_t, noise, t = self.diffuse(images)
            
            # Predict noise
            predicted_noise = self([x_t, t], training=True)
            
            # Loss is MSE between actual and predicted noise
            loss = self.compiled_loss(noise, predicted_noise)
            
        # Get gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(noise, predicted_noise)
        return {m.name: m.result() for m in self.metrics}
    
    def generate(self, num_images):
        """Generates images using the reverse diffusion process."""
        # Start with pure noise
        x_t = tf.random.normal((num_images, self.img_size, self.img_size, self.img_channels))
        
        # Gradually denoise
        for t in reversed(range(self.timesteps)):
            timesteps = tf.ones((num_images,), dtype=tf.int32) * t
            
            # Predict noise
            predicted_noise = self([x_t, timesteps], training=False)
            
            # Get schedule values for current timestep
            alpha = self.diffusion_schedule['alphas'][t]
            alpha_cumprod = self.diffusion_schedule['alphas_cumprod'][t]
            alpha_cumprod_prev = self.diffusion_schedule['alphas_cumprod_prev'][t]
            beta = self.diffusion_schedule['betas'][t]
            
            # Compute variance of q(x_{t-1} | x_t)
            variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
            
            # Sample from posterior
            if t > 0:
                noise = tf.random.normal(shape=tf.shape(x_t))
            else:
                noise = 0
                
            x_t = (
                1 / tf.sqrt(alpha) * (
                    x_t - 
                    (beta / (tf.sqrt(1 - alpha_cumprod))) * predicted_noise
                ) + tf.sqrt(variance) * noise
            )
            
        # Scale to [0, 1]
        x_t = (x_t + 1) / 2
        x_t = tf.clip_by_value(x_t, 0, 1)
        
        return x_t
