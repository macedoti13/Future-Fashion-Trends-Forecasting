from ultralytics import YOLO
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
import os
import torch

# make sure the images directory exists
if not os.path.exists("images/segmented_images"):
    os.makedirs("images/segmented_images")

HOME = os.getcwd()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join(HOME, "models", "sam_weights.pth")
IMAGE_PATH = 'images/original_images/1.jpg'

# creates YOLO model
yolo = YOLO("models/yolo_weights.pt")

# creates SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

# bbox converison function
def convert_bbox_x1y1x2y2_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    x = x1
    y = y1
    return x, y, w, h


yolo_output = yolo.predict(IMAGE_PATH, conf=0.5)

r = []
for result in yolo_output:
    for bbox in result.boxes.data,:
        box = bbox.int().cpu().numpy()
        for b in box:
            x, y, w, h = convert_bbox_x1y1x2y2_to_xywh(b[0], b[1], b[2], b[3])
            r.append([b[0], b[1], b[2], b[3], b[5]])

names = yolo_output[0].names

# Create the image variable and box_annotator
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
box_annotator = sv.BoxAnnotator(color=sv.Color.red())
mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
output = np.zeros_like(image)

for i, box in enumerate(r):
    label = box[-1]
    box = box[:-1]
    box = np.array(box)

    mask_predictor.set_image(image)
    
    masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)

    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
    detections = detections[detections.area == np.max(detections.area)]

    source_image = box_annotator.annotate(scene=image.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    # Combine the masks with a logical OR operation within the loop
    for m in masks:
        mask_combined = np.logical_or(mask_combined, m)
        
# Use the combined mask to select the pixels of the original image
output[mask_combined] = image[mask_combined]

# Save the image with the original colors
save_path = f"images/segmented_images/outfit_{os.path.basename(IMAGE_PATH).replace('.jpg', '.png')}"
cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
