import tensorflow as tf
import matplotlib.pyplot as plt
from models.diffusion import DiffusionModel

def plot_images(images, title=None):
    """Helper to plot a grid of images."""
    fig = plt.figure(figsize=(8, 8))
    rows = cols = int(len(images) ** 0.5)
    
    for idx, image in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image)
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.show()

def main():
    # Load and preprocess data
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 127.5 - 1  # Scale to [-1, 1]
    x_test = x_test.astype('float32') / 127.5 - 1
    
    # Create and compile model
    model = DiffusionModel(
        timesteps=1000,
        img_size=32,
        img_channels=3,
        embedding_dims=256
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    # Train the model
    batch_size = 64
    model.fit(
        x_train,
        epochs=50,
        batch_size=batch_size,
        validation_data=x_test,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'diffusion_model',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Generate some samples
    samples = model.generate(num_images=16)
    plot_images(samples, title='Generated Images')

if __name__ == '__main__':
    main()
