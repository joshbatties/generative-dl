import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoders import VariationalAutoencoder
from models.gans import DCGAN
from models.diffusion import DiffusionModel
from utils.data import load_and_preprocess_images

def test_vae():
    """Test VAE model."""
    print("\nTesting VAE...")
    
    # Create small test dataset
    x = tf.random.normal((4, 28, 28, 1))
    
    # Create model
    vae = VariationalAutoencoder(input_shape=(28, 28, 1), latent_dim=2)
    vae.compile(optimizer='adam')
    
    # Test forward pass
    reconstructed = vae(x)
    assert reconstructed.shape == x.shape, f"Expected shape {x.shape}, got {reconstructed.shape}"
    
    # Test training step
    loss = vae.train_step(x)
    assert 'loss' in loss, "Training step should return loss"
    print("VAE test passed!")

def test_gan():
    """Test GAN model."""
    print("\nTesting GAN...")
    
    # Create model
    gan = DCGAN(latent_dim=100, image_shape=(28, 28, 1))
    gan.compile()
    
    # Test generator
    noise = tf.random.normal((4, 100))
    generated = gan.generator(noise, training=False)
    assert generated.shape == (4, 28, 28, 1), f"Expected shape (4, 28, 28, 1), got {generated.shape}"
    
    # Test discriminator
    scores = gan.discriminator(generated, training=False)
    assert scores.shape == (4, 1), f"Expected shape (4, 1), got {scores.shape}"
    
    print("GAN test passed!")

def test_diffusion():
    """Test Diffusion model."""
    print("\nTesting Diffusion...")
    
    # Create model with smaller size for testing
    model = DiffusionModel(
        img_size=28,
        img_channels=1,
        base_channels=32,
        timesteps=100
    )
    model.compile(optimizer='adam')
    
    # Test forward pass
    x = tf.random.normal((2, 28, 28, 1))
    t = tf.constant([0, 50])
    noise_pred = model([x, t], training=False)
    assert noise_pred.shape == x.shape, f"Expected shape {x.shape}, got {noise_pred.shape}"
    
    # Test training step
    loss = model.train_step(x)
    assert 'loss' in loss, "Training step should return loss"
    
    # Test sampling
    samples = model.generate(batch_size=2)
    assert samples.shape == (2, 28, 28, 1), f"Expected shape (2, 28, 28, 1), got {samples.shape}"
    
    print("Diffusion test passed!")

def test_data_loading():
    """Test data loading utilities."""
    print("\nTesting data loading...")
    
    x_train, x_test = load_and_preprocess_images("mnist")
    assert len(x_train.shape) == 4, "Training data should be 4D"
    assert x_train.dtype == tf.float32, "Data should be float32"
    assert x_train.max() <= 1.0 and x_train.min() >= 0.0, "Data should be normalized to [0, 1]"
    
    print("Data loading test passed!")

def main():
    """Run all tests."""
    print("Starting tests...")
    
    try:
        # Test data loading
        test_data_loading()
        
        # Test models
        test_vae()
        test_gan()
        test_diffusion()
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
