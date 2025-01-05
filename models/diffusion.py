import tensorflow as tf
import numpy as np

class TimestepEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
    def call(self, timesteps):
        half = self.embedding_dim // 2
        freqs = tf.exp(
            -tf.math.log(float(self.max_period)) * 
            tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.cast(timesteps, dtype=tf.float32)[:, None] * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if self.embedding_dim % 2:
            embedding = tf.pad(embedding, [[0, 0], [0, 1]])
        return embedding

def get_beta_schedule(num_diffusion_steps, schedule_type='cosine'):
    if schedule_type == 'linear':
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_steps)
    
    elif schedule_type == 'cosine':
        steps = num_diffusion_steps + 1
        x = np.linspace(0, num_diffusion_steps, steps)
        alphas_cumprod = np.cos(((x / num_diffusion_steps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, use_conv=False, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.use_conv = use_conv
        
        self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        
        self.time_emb = tf.keras.layers.Dense(out_channels)
        
        self.norm1 = tf.keras.layers.GroupNormalization(groups=8)
        self.norm2 = tf.keras.layers.GroupNormalization(groups=8)
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        if use_conv:
            self.shortcut = tf.keras.layers.Conv2D(out_channels, 1)
        
    def call(self, x, time_emb, training=False):
        h = self.norm1(x)
        h = tf.nn.swish(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = tf.nn.swish(time_emb)
        time_emb = self.time_emb(time_emb)
        h = h + time_emb[:, None, None]
        
        h = self.norm2(h)
        h = tf.nn.swish(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        
        if self.use_conv:
            x = self.shortcut(x)
            
        return x + h

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = tf.keras.layers.GroupNormalization(groups=8)
        self.qkv = tf.keras.layers.Conv2D(channels * 3, 1)
        self.proj = tf.keras.layers.Conv2D(channels, 1)
        
    def call(self, x):
        B, H, W, C = tf.shape(x)
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = tf.reshape(qkv, (B, H * W, 3, C))
        q, k, v = tf.unstack(qkv, axis=2)
        
        scale = 1 / tf.sqrt(tf.cast(C, tf.float32))
        attention = tf.matmul(q, k, transpose_b=True) * scale
        attention = tf.nn.softmax(attention, axis=-1)
        
        h = tf.matmul(attention, v)
        h = tf.reshape(h, (B, H, W, C))
        return x + self.proj(h)

class DiffusionModel(tf.keras.Model):
    def __init__(
        self,
        img_size=32,
        img_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        timesteps=1000,
        schedule_type='cosine'
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.timesteps = timesteps
        
        # Time embedding
        time_embed_dim = base_channels * 4
        self.time_embed = tf.keras.Sequential([
            TimestepEmbedding(time_embed_dim),
            tf.keras.layers.Dense(time_embed_dim),
            tf.keras.layers.Dense(time_embed_dim),
        ])
        
        # Initialize diffusion parameters
        betas = get_beta_schedule(timesteps, schedule_type)
        self.betas = tf.cast(betas, tf.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], [[1, 0]], constant_values=1.0)
        
        # Create U-Net layers
        self.input_blocks = []
        self.middle_block = []
        self.output_blocks = []
        
        # Initial projection
        channels = base_channels
        self.conv_in = tf.keras.layers.Conv2D(channels, 3, padding='same')
        
        # Downsampling
        num_resolutions = len(channel_mults)
        current_res = img_size
        
        for level in range(num_resolutions):
            out_channels = base_channels * channel_mults[level]
            
            for _ in range(num_res_blocks):
                block = [
                    ResidualBlock(out_channels),
                    AttentionBlock(out_channels) if current_res <= 16 else tf.keras.layers.Lambda(lambda x: x)
                ]
                self.input_blocks.append(block)
                channels = out_channels
                
            if level != num_resolutions - 1:
                self.input_blocks.append([tf.keras.layers.AveragePooling2D()])
                current_res = current_res // 2
        
        # Middle
        self.middle_block = [
            ResidualBlock(channels),
            AttentionBlock(channels),
            ResidualBlock(channels),
        ]
        
        # Upsampling
        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * channel_mults[level]
            
            for _ in range(num_res_blocks + 1):
                block = [
                    ResidualBlock(out_channels),
                    AttentionBlock(out_channels) if current_res <= 16 else tf.keras.layers.Lambda(lambda x: x)
                ]
                self.output_blocks.append(block)
                channels = out_channels
                
            if level != 0:
                self.output_blocks.append([tf.keras.layers.UpSampling2D()])
                current_res = current_res * 2
        
        self.conv_out = tf.keras.Sequential([
            tf.keras.layers.GroupNormalization(groups=8),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(img_channels, 3, padding='same'),
        ])
        
    def call(self, inputs, training=False):
        x, t = inputs
        
        # Time embedding
        t = self.time_embed(t)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling
        hs = [h]
        for blocks in self.input_blocks:
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t, training=training)
                elif isinstance(block, AttentionBlock):
                    h = block(h)
                else:
                    h = block(h)
            hs.append(h)
        
        # Middle
        for block in self.middle_block:
            if isinstance(block, ResidualBlock):
                h = block(h, t, training=training)
            else:
                h = block(h)
        
        # Upsampling
        for blocks in self.output_blocks:
            h = tf.concat([h, hs.pop()], axis=-1)
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t, training=training)
                elif isinstance(block, AttentionBlock):
                    h = block(h)
                else:
                    h = block(h)
        
        # Output
        return self.conv_out(h)
    
    @tf.function
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        
        # Sample timesteps uniformly
        t = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self.timesteps,
            dtype=tf.int32
        )
        
        with tf.GradientTape() as tape:
            # Generate noise
            noise = tf.random.normal(tf.shape(data))
            
            # Get noise schedule parameters for timestep t
            alpha_cumprod_t = tf.gather(self.alphas_cumprod, t)
            alpha_cumprod_t = tf.reshape(alpha_cumprod_t, (-1, 1, 1, 1))
            
            # Add noise according to schedule
            noisy_data = (
                tf.sqrt(alpha_cumprod_t) * data +
                tf.sqrt(1 - alpha_cumprod_t) * noise
            )
            
            # Predict noise
            predicted_noise = self([noisy_data, t], training=True)
            
            # Calculate loss
            loss = tf.reduce_mean(tf.square(noise - predicted_noise))
        
        # Update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': loss}
    
    def diffusion_schedule(self, n_steps):
        """Returns diffusion schedule parameters for n_steps."""
        times = np.arange(n_steps)
        alphas = self.alphas
        alpha_cumprod = self.alphas_cumprod
        alpha_cumprod_prev = self.alphas_cumprod_prev
        betas = self.betas
        
        return {
            'times': times,
            'alphas': alphas,
            'alpha_cumprod': alpha_cumprod,
            'alpha_cumprod_prev': alpha_cumprod_prev,
            'betas': betas
        }
    
    @tf.function
    def generate(self, batch_size):
        """Generates images using the reverse diffusion process."""
        # Start from pure noise
        x = tf.random.normal((batch_size, self.img_size, self.img_size, self.img_channels))
        
        # Iterate through timesteps backwards
        for t in tf.range(self.timesteps - 1, -1, -1):
            t_batched = tf.fill([batch_size], t)
            
            # Predict noise
            predicted_noise = self([x, t_batched], training=False)
            
            # Get schedule parameters
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            beta = self.betas[t]
            
            # Only add noise if t > 0
            if t > 0:
                noise = tf.random.normal(tf.shape(x))
            else:
                noise = 0
            
            # Update x using reverse diffusion formula
            x = (
                1 / tf.sqrt(alpha) *
                (x - (beta / (tf.sqrt(1 - alpha_cumprod))) * predicted_noise)
                + tf.sqrt(beta) * noise
            )
        
        return x
