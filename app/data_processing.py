import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

IMG_SIZE = (148, 148)


def tf_load_image_float01(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


def _apply_text_watermark_np(img_float32: np.ndarray, text: str = "HI MY NAME IS ARTEM",
                             opacity: float = 0.2) -> np.ndarray:

    if isinstance(text, (bytes, np.ndarray, np.bytes_)):
        text = text.decode('utf-8') if hasattr(text, 'decode') else str(text)

    img_float32 = np.clip(img_float32, 0.0, 1.0)
    img_uint8 = (img_float32 * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="RGB").convert("RGBA")

    txt_layer = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    color = (255, 255, 255, int(255 * opacity))

    width, height = pil_img.size
    step_x = 100
    step_y = 50

    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            offset = 50 if (y // step_y) % 2 == 0 else 0
            draw.text((x + offset, y), text, fill=color, font=font)

    out = Image.alpha_composite(pil_img, txt_layer).convert("RGB")
    return np.array(out).astype(np.float32) / 255.0


def tf_apply_watermark(image_tensor, text="HI MY NAME IS ARTEM"):
    """Обертка для использования в tf.data.Dataset"""
    return tf.numpy_function(
        _apply_text_watermark_np,
        [image_tensor, text],
        tf.float32
    )


def make_dataset(paths, batch_size: int = 32, training: bool = True,
                 wm_text: str = "academy.ai") -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)

    if training:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    def fn(path):
        clean = tf_load_image_float01(path)

        if training:
            clean = _augmenter(clean, training=True)
            clean = tf_augment(clean)

        wm = tf_apply_watermark(clean, text=wm_text)

        return wm, clean

    return (
        ds
        .map(fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_augmenter(seed: int = 42) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="reflect",
            seed=seed,
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.08, width_factor=0.08,
            fill_mode="reflect", seed=seed,
        ),
        tf.keras.layers.RandomBrightness(factor=0.20, seed=seed),
        tf.keras.layers.RandomContrast(factor=0.20, seed=seed),
    ], name="augmenter")


@tf.function
def tf_augment(img: tf.Tensor) -> tf.Tensor:

    img = tf.cond(
        tf.random.uniform([]) < 0.5,
        lambda: tf.image.random_saturation(img, lower=0.6, upper=1.4),
        lambda: img,
    )

    img = tf.cond(
        tf.random.uniform([]) < 0.4,
        lambda: tf.cast(
            tf.image.random_jpeg_quality(
                tf.image.convert_image_dtype(img, tf.uint8),
                min_jpeg_quality=65, max_jpeg_quality=95,
            ), tf.float32) / 255.0,
        lambda: img,
    )

    return tf.clip_by_value(img, 0.0, 1.0)


_augmenter = build_augmenter()
