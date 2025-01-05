import tensorflow as tf
from models.autoencoders import VariationalAutoencoder
from utils.data import prepare_dataset
from utils.visualization import plot_reconstructions, plot_latent_space

def main():
    # Load dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    
    # Prepare data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    # Create dataset
    train_dataset = prepare_dataset(x_train, batch_size=32)
    test_dataset = prepare_dataset(x_test, batch_size=32)
    
    # Create model
    vae = VariationalAutoencoder(
        input_shape=(28, 28, 1),
        latent_dim=2  # Small latent dimension for visualization
    )
    
    # Compile and train
    vae.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )
    
    history = vae.model.fit(
        train_dataset,
        epochs=30,
        validation_data=test_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Generate reconstructions
    sample_images = next(iter(test_dataset))[0][:10]
    reconstructions = vae.model.predict(sample_images)
    plot_reconstructions(sample_images, reconstructions)
    
    # Generate samples from latent space
    plot_latent_space(vae, n=20)

if __name__ == "__main__":
    main()
