import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np

# Add parent directory to path to import from models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import DiffusionModel
from utils.data import load_and_preprocess_images, create_training_datasets

def plot_diffusion_process(model, image, num_steps=8):
    """Visualize forward diffusion process on a single image."""
    fig, ax = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
    
    # Get timesteps
    timesteps = np.linspace(0, model.timesteps-1, num_steps).astype(int)
    
    # Create batch dimension
    image_batch = tf.expand_dims(image, 0)
    
    for idx, t in enumerate(timesteps):
        # Get noise schedule parameters
        alpha_cumprod = tf.gather(model.alphas_cumprod, t)
        alpha_cumprod = tf.reshape(alpha_cumprod, (-1, 1, 1, 1))
        
        # Add noise according to schedule
        noise = tf.random.normal(tf.shape(image_batch))
        noisy_image = (
            tf.sqrt(alpha_cumprod) * image_batch +
            tf.sqrt(1 - alpha_cumprod) * noise
        )
        
        # Plot
        ax[idx].imshow(noisy_image[0].numpy(), cmap='gray')
        ax[idx].axis('off')
        ax[idx].set_title(f't={t}')
    
    plt.tight_layout()
    return fig

def save_samples(model, epoch, num_samples=16, output_dir="samples"):
    """Generate and save samples from the model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    samples = model.generate(num_samples)
    
    # Create grid of images
    rows = int(np.sqrt(num_samples))
    cols = num_samples // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(samples[idx].numpy(), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'samples_epoch_{epoch:04d}.png'))
    plt.close()

def main():
    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    TIMESTEPS = 1000
    BASE_CHANNELS = 64
    SAVE_EVERY = 10
    OUTPUT_DIR = "diffusion_samples"
    
    # Load and prepare data
    x_train, x_test = load_and_preprocess_images("fashion_mnist")
    train_dataset, val_dataset, test_dataset = create_training_datasets(
        x_train, x_test, batch_size=BATCH_SIZE
    )
    
    # Create model
    model = DiffusionModel(
        img_size=28,
        img_channels=1,
        base_channels=BASE_CHANNELS,
        timesteps=TIMESTEPS
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4)
    )
    
    # Visualize forward diffusion on a sample image
    sample_image = next(iter(test_dataset))[0]
    fig = plot_diffusion_process(model, sample_image)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, 'forward_diffusion.png'))
    plt.close()
    
    # Training loop
    print("Starting training...")
    losses = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        start_time = time.time()
        
        # Train for one epoch
        epoch_losses = []
        for batch in train_dataset:
            loss = model.train_step(batch)
            epoch_losses.append(loss['loss'].numpy())
        
        # Calculate average loss
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Print progress
        time_taken = time.time() - start_time
        print(f"Loss: {avg_loss:.4f}")
        print(f"Time taken: {time_taken:.2f}s")
        
        # Save samples periodically
        if (epoch + 1) % SAVE_EVERY == 0:
            print("Generating samples...")
            save_samples(model, epoch + 1, output_dir=OUTPUT_DIR)
            
            # Save model weights
            model.save_weights(os.path.join(OUTPUT_DIR, f'diffusion_model_epoch_{epoch+1}'))
            
            # Plot loss curve
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
            plt.close()
    
    # Generate final samples
    print("\nGenerating final samples...")
    save_samples(model, EPOCHS, num_samples=25, output_dir=OUTPUT_DIR)
    
    # Save final model
    model.save_weights(os.path.join(OUTPUT_DIR, 'diffusion_model_final'))
    
    # Plot final loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_training_loss.png'))
    plt.close()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
