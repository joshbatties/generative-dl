import tensorflow as tf
import numpy as np
from scipy import linalg
import tensorflow_hub as hub

def calculate_fid(real_images, generated_images, inception_model=None):
    """Calculate Fréchet Inception Distance between real and generated images."""
    if inception_model is None:
        inception_model = tf.keras.applications.InceptionV3(
            include_top=False,
            pooling='avg',
            input_shape=(299, 299, 3)
        )
    
    def preprocess_images(images):
        # Resize to inception size
        images = tf.image.resize(images, (299, 299))
        # Convert grayscale to RGB if needed
        if images.shape[-1] == 1:
            images = tf.image.grayscale_to_rgb(images)
        # Preprocess for inception
        images = tf.keras.applications.inception_v3.preprocess_input(images)
        return images
    
    # Get activations
    real_images = preprocess_images(real_images)
    generated_images = preprocess_images(generated_images)
    
    real_activations = inception_model.predict(real_images)
    generated_activations = inception_model.predict(generated_images)
    
    # Calculate mean and covariance
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

def calculate_inception_score(images, inception_model=None, splits=10):
    """Calculate Inception Score for generated images."""
    if inception_model is None:
        inception_model = tf.keras.applications.InceptionV3(include_top=True)
    
    def get_predictions(processed_images):
        preds = inception_model.predict(processed_images)
        return preds
    
    # Preprocess images
    processed_images = tf.image.resize(images, (299, 299))
    if processed_images.shape[-1] == 1:
        processed_images = tf.image.grayscale_to_rgb(processed_images)
    processed_images = tf.keras.applications.inception_v3.preprocess_input(processed_images)
    
    # Get predictions
    preds = get_predictions(processed_images)
    
    # Calculate scores for splits
    scores = []
    for i in range(splits):
        part = preds[(i * len(preds) // splits):((i + 1) * len(preds) // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)

def calculate_reconstruction_error(original_images, reconstructed_images, metric='mse'):
    """Calculate reconstruction error using various metrics."""
    if metric == 'mse':
        return tf.reduce_mean(tf.square(original_images - reconstructed_images))
    elif metric == 'mae':
        return tf.reduce_mean(tf.abs(original_images - reconstructed_images))
    elif metric == 'rmse':
        return tf.sqrt(tf.reduce_mean(tf.square(original_images - reconstructed_images)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def ssim_score(original_images, generated_images):
    """Calculate Structural Similarity Index (SSIM) between image sets."""
    ssim_values = []
    for orig, gen in zip(original_images, generated_images):
        ssim = tf.image.ssim(
            orig,
            gen,
            max_val=1.0
        )
        ssim_values.append(ssim)
    return tf.reduce_mean(ssim_values)

def psnr_score(original_images, generated_images):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between image sets."""
    psnr_values = []
    for orig, gen in zip(original_images, generated_images):
        psnr = tf.image.psnr(
            orig,
            gen,
            max_val=1.0
        )
        psnr_values.append(psnr)
    return tf.reduce_mean(psnr_values)
