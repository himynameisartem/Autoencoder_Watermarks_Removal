import cv2
import numpy as np


def apply_text_watermark(image_float01: np.ndarray, text="hi there") -> np.ndarray:

    img = (image_float01 * 255).astype(np.uint8)

    overlay = img.copy()

    h, w = img.shape[:2]

    cv2.putText(
        overlay,
        text,
        (w // 4, h // 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        text,
        (w // 4, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    return result.astype(np.float32) / 255.0
