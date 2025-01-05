import tensorflow as tf
import numpy as np
from scipy import linalg

def compute_reconstruction_error(original, reconstructed, metric='mse'):
    """Compute reconstruction error using various metrics."""
    if metric == 'mse':
        return tf.reduce_mean(tf.square(original - reconstructed))
    elif metric == 'mae':
        return tf.reduce_mean(tf.abs(original - reconstructed))
    elif metric == 'rmse':
        return tf.sqrt(tf.reduce_mean(tf.square(original - reconstructed)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def calculate_fid(real_features, generated_features):
    """Calculate Fréchet Inception Distance between real and generated images."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

class InceptionScore:
    """Calculate Inception Score for generated images."""
    def __init__(self, model=None):
        if model is None:
            self.model = tf.keras.applications.InceptionV3(
                include_top=True,
                weights='imagenet',
                input_shape=(299, 299, 3)
            )
    
    def calculate_score(self, images, batch_size=32, splits=10):
        """Calculate inception score for given images."""
        # Resize images to inception size
        processed_images = tf.image.resize(images, (299, 299))
        processed_images = tf.keras.applications.inception_v3.preprocess_input(
            processed_images * 255
        )
        
        # Get predictions
        preds = []
        n_batches = len(images) // batch_size
        for i in range(n_batches):
            pred = self.model.predict(
                processed_images[i * batch_size:(i + 1) * batch_size]
            )
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)
        
        # Calculate score
        scores = []
        for i in range(splits):
            part = preds[
                (i * len(preds) // splits):((i + 1) * len(preds) // splits), :
            ]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)

def wasserstein_loss(y_true, y_pred):
    """Wasserstein loss for WGAN."""
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_samples, fake_samples):
    """Gradient penalty for WGAN-GP."""
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_samples - real_samples
    interpolated = real_samples + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = gp_tape.gradient(pred, interpolated)[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
