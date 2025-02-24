import cv2
import numpy as np

def draw_predictions(image, results):
    """Draws bounding boxes, masks, and labels on an image."""
    for result in results:
        # Draw bounding boxes for cars and persons
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]

            # Skip drivable_area and lane classes (they are handled as masks)
            if class_name in ["drivable area", "lane"]:
                continue

            # Assign different colors for different classes
            if class_name == "car":
                color = (0, 255, 0)  # Green for cars
            elif class_name == "person":
                color = (255, 0, 0)  # Blue for persons
            else:
                color = (255, 255, 0)  # Yellow for other objects

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw masks for drivable areas and lanes
        if hasattr(result, "masks") and result.masks is not None:
            for mask, class_id in zip(result.masks, result.boxes.cls):
                class_name = result.names[int(class_id)]

                # Only draw masks for drivable areas and lanes
                if class_name not in ["drivable area", "lane"]:
                    continue

                # Assign strong colors for drivable areas and lanes
                if class_name == "drivable area":
                    overlay_color = (0, 0, 255)  # Solid Red for drivable areas
                elif class_name == "lane":
                    overlay_color = (255, 0, 255)  # Solid Magenta for lanes

                # Convert mask to numpy array and resize it
                mask = mask.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)  # Scale to [0, 255]
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

                # Create a colored overlay
                overlay = np.zeros_like(image, dtype=np.uint8)
                overlay[mask > 0] = overlay_color

                # Blend the overlay with the original image
                alpha = 0.5  # Opacity of the overlay
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image


