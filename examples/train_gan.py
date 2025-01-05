import tensorflow as tf
from models.gans import DCGAN
from utils.data import prepare_dataset
from utils.visualization import plot_gan_training
from utils.metrics import calculate_fid
import numpy as np

def main():
    # Load and prepare dataset
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_dataset = prepare_dataset(
        x_train,
        batch_size=64,
        image_size=(28, 28)
    )

    # Create GAN
    gan = DCGAN(
        latent_dim=100,
        image_shape=(28, 28, 1)
    )

    # Training parameters
    epochs = 100
    gen_losses = []
    disc_losses = []

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in train_dataset:
            # Train discriminator
            noise = tf.random.normal([64, gan.latent_dim])
            generated_images = gan.generator(noise, training=False)
            real_loss = gan.discriminator.train_on_batch(batch, tf.ones((64, 1)))
            fake_loss = gan.discriminator.train_on_batch(generated_images, tf.zeros((64, 1)))
            disc_loss = 0.5 * (real_loss[0] + fake_loss[0])
            
            # Train generator
            noise = tf.random.normal([64, gan.latent_dim])
            gen_loss = gan.model.train_on_batch(noise, tf.ones((64, 1)))
            
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
        
        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([16, gan.latent_dim])
            generated_images = gan.generator.predict(noise)
            
            # Save samples
            filename = f"samples/epoch_{epoch+1}.png"
            tf.keras.preprocessing.image.save_img(
                filename,
                tf.concat([img for img in generated_images[:16]], axis=1)
            )
    
    # Plot training progression
    plot_gan_training(gen_losses, disc_losses)
    
    # Generate final samples
    noise = tf.random.normal([100, gan.latent_dim])
    generated_images = gan.generator.predict(noise)
    
    # Calculate FID score
    real_features = gan.discriminator.predict(next(iter(train_dataset)))
    generated_features = gan.discriminator.predict(generated_images)
    fid = calculate_fid(real_features, generated_features)
    print(f"Final FID score: {fid}")

if __name__ == "__main__":
    main()
