# Generative Deep Learning Models

A comprehensive implementation of modern generative deep learning architectures, including Autoencoders, Generative Adversarial Networks (GANs), and Diffusion Models. Built with TensorFlow/Keras, this repository provides clean, modular implementations along with training scripts and examples.

## 🚀 Features

- **Multiple Model Architectures:**
  - Basic and Convolutional Autoencoders
  - Variational Autoencoders (VAEs)
  - Deep Convolutional GANs (DCGANs)
  - Wasserstein GANs (WGANs)
  - Denoising Diffusion Probabilistic Models (DDPM)

- **Key Capabilities:**
  - Image generation
  - Latent space manipulation
  - Feature extraction
  - Image reconstruction
  - Progressive denoising

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/generative-dl
cd generative-dl

# Install dependencies
pip install -r requirements.txt
```

## 💻 Quick Start

### Training a Diffusion Model

```python
from models.diffusion import DiffusionModel
import tensorflow as tf

# Load dataset (e.g., CIFAR-10)
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 127.5 - 1  # Scale to [-1, 1]

# Create and train model
model = DiffusionModel(
    timesteps=1000,
    img_size=32,
    img_channels=3,
    embedding_dims=256
)

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, epochs=50, batch_size=64)

# Generate new images
samples = model.generate(num_images=16)
```

### Variational Autoencoder Example

```python
from models.autoencoders import VariationalAutoencoder

# Initialize VAE
vae = VariationalAutoencoder(
    input_shape=(28, 28, 1),
    latent_dim=32
)

# Train
vae.model.compile(optimizer='adam')
vae.model.fit(x_train, x_train, epochs=20, batch_size=128)

# Generate new images
z = tf.random.normal((16, 32))  # Random latent vectors
generated = vae.decoder.predict(z)
```

### DCGAN Training

```python
from models.gans import DCGAN

# Create GAN
dcgan = DCGAN(
    latent_dim=100,
    image_shape=(28, 28, 1)
)

# Train
dcgan.train(
    dataset=train_dataset,
    epochs=100,
    batch_size=32
)
```

## 📚 Model Details

### Diffusion Models
- Implementation of DDPM with improved sampling
- Cosine noise schedule
- U-Net architecture with timestep embedding
- Progressive denoising process

### Autoencoders
- Basic, Convolutional, and Variational implementations
- KL divergence regularization for VAEs
- Flexible latent dimensions
- Optional denoising capability

### GANs
- DCGAN architecture with batch normalization
- Wasserstein loss option with gradient penalty
- Progressive growing capability
- Spectral normalization for stability

## 🛠️ Advanced Usage

### Custom Training Loop

```python
# Example of custom training for GANs
@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, latent_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Update weights
    generator_gradients = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)
        
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables))
```

## 📈 Results

Example generations from each model type:

### DDPM Samples
[Results coming soon]

### VAE Reconstructions
[Results coming soon]

### GAN Generations
[Results coming soon]

## 📖 Documentation

Detailed documentation for each model and component is available in the `docs/` directory:
- [Diffusion Models](docs/diffusion.md)
- [Autoencoders](docs/autoencoders.md)
- [GANs](docs/gans.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Citations

If you use this code in your research, please cite:

```bibtex
@misc{ddpm2020,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  year={2020},
  eprint={2006.11239},
  archivePrefix={arXiv}
}
```

## 🙏 Acknowledgments

- DDPM implementation inspired by the original paper by Ho et al.
- VAE architecture based on Auto-Encoding Variational Bayes
- GAN implementations following DCGAN and WGAN papers

## ⚠️ Notes

- The models are implemented with TensorFlow 2.x
- GPU is recommended for training diffusion models
- Some architectures may need adjustment for different image sizes
- Hyperparameters may need tuning for specific datasets
