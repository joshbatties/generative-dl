import tensorflow as tf
import numpy as np

def prepare_dataset(dataset, batch_size, buffer_size=10000, image_size=None):
    """Prepare a dataset for training generative models."""
    if isinstance(dataset, tuple):
        # If dataset is a tuple of (images, labels), take only images
        images = dataset[0]
    else:
        images = dataset
        
    # Convert to float32 and scale to [-1, 1]
    images = tf.cast(images, tf.float32)
    images = (images - 127.5) / 127.5
    
    # Resize if needed
    if image_size is not None:
        images = tf.image.resize(images, image_size)
    
    # Create tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def add_noise(images, noise_factor=0.3):
    """Add random noise to images."""
    noisy_images = images + noise_factor * tf.random.normal(tf.shape(images))
    return tf.clip_by_value(noisy_images, 0.0, 1.0)

class DiffusionDataset:
    """Dataset wrapper for diffusion models."""
    def __init__(self, dataset, timesteps, batch_size=32):
        self.dataset = dataset
        self.timesteps = timesteps
        self.batch_size = batch_size
        
    def prepare_training_batch(self, images):
        """Prepare a batch of data for training diffusion models."""
        batch_size = tf.shape(images)[0]
        
        # Sample random timesteps
        t = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self.timesteps,
            dtype=tf.int32
        )
        
        # Sample random noise
        noise = tf.random.normal(tf.shape(images))
        
        # Get noise schedule parameters
        alpha_bar = self._get_alpha_bar(t)
        
        # Add noise according to schedule
        noisy_images = (
            tf.sqrt(alpha_bar)[:, None, None, None] * images + 
            tf.sqrt(1 - alpha_bar)[:, None, None, None] * noise
        )
        
        return noisy_images, noise, t
    
    def _get_alpha_bar(self, t):
        """Get cumulative product of (1-beta) for given timesteps."""
        # This is a simplified version - actual implementation would use
        # the noise schedule from your diffusion model
        beta = tf.linspace(1e-4, 0.02, self.timesteps)
        alpha = 1. - beta
        alpha_bar = tf.math.cumprod(alpha)
        return tf.gather(alpha_bar, t)

def create_image_grid(images, rows, cols):
    """Create a grid of images for visualization."""
    assert len(images) >= rows * cols
    
    # Reshape and transpose for grid
    grid = np.zeros((rows * images.shape[1], cols * images.shape[2], images.shape[3]))
    for idx in range(rows * cols):
        i = idx % cols
        j = idx // cols
        grid[j*images.shape[1]:(j+1)*images.shape[1], 
             i*images.shape[2]:(i+1)*images.shape[2]] = images[idx]
    
    return grid
