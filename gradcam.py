"""Grad-CAM for the nested DenseNet121 model.

Target layer: conv5_block16_concat (last conv output of the DenseNet-121 backbone).
The backbone is nested inside the outer Functional model as model.get_layer('densenet121').
"""
import numpy as np
import tensorflow as tf
from PIL import Image


def build_grad_model(model: tf.keras.Model):
    """Build extractor for CAM layer output and prediction path.
    
    The DenseNet121 backbone is nested inside the top-level Functional model.
    In Keras 3, we extract via the backbone's intermediate output and trace through
    the outer model layers.
    """
    import keras
    base = model.get_layer("densenet121")
    cam_layer = base.get_layer("conv5_block16_concat")
    base_grad_model = keras.Model(inputs=base.input, outputs=[cam_layer.output, base.output])
    base_idx = model.layers.index(base)
    layers_before = model.layers[:base_idx]
    layers_after = model.layers[base_idx + 1:]
    return (base_grad_model, layers_before, layers_after)


def compute_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    grad_model=None,
) -> np.ndarray:
    """Return a 384x384 heatmap in [0, 1] for the predicted class.

    Parameters
    ----------
    model : the outer Keras model (used only if grad_model is None).
    img_array : float32 array shape (1, 384, 384, 3), raw pixels [0, 255].
    grad_model : pre-built grad extractor tuple from build_grad_model.

    Returns
    -------
    cam : ndarray shape (384, 384) in [0, 1].
    """
    if grad_model is None:
        grad_model = build_grad_model(model)

    base_grad_model, layers_before, layers_after = grad_model
    img_tensor = tf.constant(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        h = img_tensor
        for l in layers_before:
            if l != model.layers[0]:  # skip input layer
                h = l(h, training=False)
        
        conv_out, base_out = base_grad_model(h, training=False)
        tape.watch(conv_out)
        
        out = base_out
        for l in layers_after:
            out = l(out, training=False)
        
        pred_class = int(tf.argmax(out, axis=1)[0])
        score = out[:, pred_class]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)
    cam = tf.nn.relu(
        tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)
    )[0].numpy()

    # Resize to input dimensions
    cam = np.array(Image.fromarray(cam).resize((384, 384), Image.BILINEAR))

    # Normalize to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam


def overlay_gradcam(
    original: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay jet-coloured Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original : PIL RGB image (any size — resized to 384x384 internally).
    cam : (384, 384) array in [0, 1].
    alpha : blending factor for the heatmap.

    Returns
    -------
    overlay : PIL RGB image 384x384.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    base = original.resize((384, 384), Image.BILINEAR)
    base_arr = np.asarray(base, dtype=np.float32) / 255.0

    jet = cm.get_cmap("jet")
    heatmap = jet(cam)[:, :, :3]  # drop alpha channel

    blended = (1 - alpha) * base_arr + alpha * heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
