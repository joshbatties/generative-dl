import tensorflow as tf
import numpy as np

def load_and_preprocess_images(dataset="mnist", image_size=None):
    """Load and preprocess image dataset.
    
    Args:
        dataset: str, one of ["mnist", "fashion_mnist", "cifar10"]
        image_size: tuple, optional target size for resizing
    
    Returns:
        train_data, test_data: Preprocessed and batched datasets
    """
    if dataset == "mnist":
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")
    elif dataset == "fashion_mnist":
        (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")
    elif dataset == "cifar10":
        (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Normalize to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Resize if needed
    if image_size is not None:
        x_train = tf.image.resize(x_train, image_size)
        x_test = tf.image.resize(x_test, image_size)
    
    return x_train, x_test

def prepare_dataset(images, batch_size=32, buffer_size=1000, shuffle=True):
    """Create a batched and prefetched dataset."""
    dataset = tf.data.Dataset.from_tensor_slices(images)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_training_datasets(x_train, x_test, batch_size=32, validation_split=0.1):
    """Create training, validation and test datasets."""
    # Create validation split
    val_size = int(len(x_train) * validation_split)
    x_val = x_train[-val_size:]
    x_train = x_train[:-val_size]
    
    # Create datasets
    train_dataset = prepare_dataset(x_train, batch_size=batch_size)
    val_dataset = prepare_dataset(x_val, batch_size=batch_size)
    test_dataset = prepare_dataset(x_test, batch_size=batch_size)
    
    return train_dataset, val_dataset, test_dataset
