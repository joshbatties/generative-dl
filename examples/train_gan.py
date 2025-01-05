import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np

# Add parent directory to path to import from models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gans import DCGAN
from utils.data import load_and_preprocess_images, create_training_datasets

def create_sample_grid(images, grid_size=4):
    """Create a grid of sample images."""
    if images.shape[-1] == 1:  # If single channel, remove the last dimension
        images = images.squeeze(-1)
    
    fig = plt.figure(figsize=(grid_size, grid_size))
    for i in range(grid_size * grid_size):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        plt.axis('off')
    return fig

def save_training_progress(gan, epoch, output_dir="samples"):
    """Generate and save sample images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    noise = tf.random.normal([16, gan.latent_dim])
    generated_images = gan.generator(noise, training=False)
    generated_images = (generated_images + 1) * 0.5  # Scale from [-1, 1] to [0, 1]
    
    # Create and save grid
    fig = create_sample_grid(generated_images.numpy())
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch:04d}.png'))
    plt.close()

def main():
    # Parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LATENT_DIM = 100
    OUTPUT_DIR = "gan_samples"
    
    # Load and prepare data
    x_train, x_test = load_and_preprocess_images("fashion_mnist")
    train_dataset, val_dataset, test_dataset = create_training_datasets(
        x_train, x_test, batch_size=BATCH_SIZE
    )
    
    # Create and compile GAN
    image_shape = (28, 28, 1)  # Fashion MNIST image shape
    gan = DCGAN(latent_dim=LATENT_DIM, image_shape=image_shape)
    gan.compile()
    
    # Training variables
    generator_losses = []
    discriminator_losses = []
    
    # Training loop
    print("Starting GAN training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        start = time.time()
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        for batch_idx, real_images in enumerate(train_dataset):
            # Train discriminator
            noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
            generated_images = gan.generator(noise, training=False)
            
            real_labels = tf.ones((BATCH_SIZE, 1))
            fake_labels = tf.zeros((BATCH_SIZE, 1))
            
            # Add noise to labels for one-sided label smoothing
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
            
            d_loss_real = gan.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = gan.discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
            g_loss = gan.gan.train_on_batch(noise, real_labels)
            
            epoch_g_losses.append(g_loss)
            epoch_d_losses.append(d_loss[0])  # Keep only the loss value, not metrics
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}: d_loss = {d_loss[0]:.4f}, g_loss = {g_loss:.4f}")
        
        # Save losses
        generator_losses.append(np.mean(epoch_g_losses))
        discriminator_losses.append(np.mean(epoch_d_losses))
        
        # Save generated images
        if (epoch + 1) % 10 == 0:
            save_training_progress(gan, epoch + 1, OUTPUT_DIR)
        
        print(f"Time for epoch {epoch+1}: {time.time()-start:.2f} sec")
        print(f"Generator loss: {generator_losses[-1]:.4f}")
        print(f"Discriminator loss: {discriminator_losses[-1]:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
    
    # Save final model
    gan.generator.save('generator.h5')
    gan.discriminator.save('discriminator.h5')
    print("\nTraining completed. Models saved.")
    
    # Generate final samples
    print("\nGenerating final samples...")
    noise = tf.random.normal([16, LATENT_DIM])
    generated_images = gan.generator(noise, training=False)
    generated_images = (generated_images + 1) * 0.5
    
    fig = create_sample_grid(generated_images.numpy())
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_samples.png'))
    plt.close()

if __name__ == "__main__":
    main()
