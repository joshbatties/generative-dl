import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_reconstructions(original_images, reconstructed_images, n=10):
    """Plot original and reconstructed images side by side."""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructed_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def plot_latent_space(model, n=30, figsize=15):
    """Plot samples from the latent space decoded through the model."""
    figure = np.zeros((figsize * n, figsize * n))
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(figsize, figsize)
            figure[i * figsize: (i + 1) * figsize,
                   j * figsize: (j + 1) * figsize] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure)
    plt.axis('off')
    plt.show()

def plot_diffusion_process(model, x_0, timesteps=10):
    """Visualize the forward diffusion process."""
    fig, axes = plt.subplots(1, timesteps, figsize=(20, 2))
    for t, ax in enumerate(axes):
        noisy_image, _, _ = model.diffuse(
            tf.expand_dims(x_0, 0), 
            t=tf.constant([int(t * model.timesteps / timesteps)])
        )
        ax.imshow(noisy_image[0])
        ax.axis('off')
        ax.set_title(f't={t * model.timesteps / timesteps:.0f}')
    plt.tight_layout()
    plt.show()

def plot_gan_training(gen_losses, disc_losses):
    """Plot GAN training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
