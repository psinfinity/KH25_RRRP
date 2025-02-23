import cv2
import numpy as np

def draw_polygons(image, results):
    """Fills polygon areas based on vertex data."""
    for result in results:
        if hasattr(result, "masks") and result.masks is not None:
            for mask in result.masks.xy:
                polygon = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(image, [polygon], color=(255, 0, 0, 100))  # Semi-transparent fill

    return image
