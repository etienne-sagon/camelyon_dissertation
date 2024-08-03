import openslide
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from torchvision import transforms
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation, median
import xml.etree.ElementTree as ET
from torchvision import models
import torch.nn as nn


import src.preprocessing.wsi_ops as ops


# Function to extract and preprocess a patch
def extract_and_preprocess_patch(slide, coords, patch_size=224, level = 0):
    x, y = coords
    patch = slide.read_region((x * patch_size, y * patch_size), level, (patch_size, patch_size)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(patch).unsqueeze(0)

def get_googlenet_model(num_classes=2, pretrained=False):

    model = models.googlenet(weights=None, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Load your GoogLeNet model
def load_model(model_path):

    DEVICE = torch.device("cpu")
    
    # Initialize the model architecture
    model = get_googlenet_model(num_classes=2, pretrained=False)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # If the state_dict was saved with DataParallel, remove the 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    

    # Load the state dict into the model   
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    print("model loaded")
    return model

# Function to make prediction for a single patch
def predict_patch(model, patch):
    with torch.no_grad():
        output = model(patch)
        probabilities = torch.softmax(output, dim=1)
    return probabilities[0, 1].item()


def parse_xml_annotation(xml_path, mag_level):
    
    """
    Parse XML annotation to get bounding boxes of the tumor regions.

    :param xml_path: Path to the XML file.
    :param mag_level: Level at which I'm reading the WSI
    :return: List of bounding boxes in (x, y, width, height) format.
    """

    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    list_annotations = {}

    mag_factor = pow(2, mag_level)

    i = 0
    for coords in root.iter('Coordinates'):
        vasc = []
        for coord in coords:
            vasc.append((int(float(coord.attrib.get("X"))/mag_factor),int(float(coord.attrib.get("Y"))/mag_factor)))
        list_annotations[i] = vasc
        i+=1
    return list_annotations

# def read_xml_annotation(xml_path):
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     annotations = []
#     for annotation in root.findall(".//Annotation"):
#         x_coords = [int(float(coord.get('X'))) for coord in annotation.findall(".//Coordinate")]
#         y_coords = [int(float(coord.get('Y'))) for coord in annotation.findall(".//Coordinate")]
#         annotations.append({
#             'xmin': min(x_coords),
#             'xmax': max(x_coords),
#             'ymin': min(y_coords),
#             'ymax': max(y_coords)
#         })
#     return annotations
    
def extract_tissue(wsi_image, level):

    """
    https://github.com/NMPoole/CS5199-Dissertation/blob/main/src/tools/4_generate_tissue_images.py

    Create a binary mask of the tissue regions using OTSU algorithm in the HSV color space.
    Create bounding boxes around the tissue regions.

    :param rgb_image: Image in an array format 
    :return binary mask: Binary mask of the image (array)
    :return bounding_boxes: list of bounding boxes
    """
    #level = utils.mag_level_tissue
    
    wsi_dims = wsi_image.level_dimensions[level]
    
    rgb_image = np.array(wsi_image.read_region((0, 0), level, wsi_dims).convert("RGB"))

    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Perform Otsu thresholding on Hue and Saturation channels
    hue_channel = hsv_image[:, :, 0]
    sat_channel = hsv_image[:, :, 1]
    otsu_thresh_hue = threshold_otsu(hue_channel)
    otsu_thresh_sat = threshold_otsu(sat_channel)
    binary_mask_hue = hue_channel <= otsu_thresh_hue
    binary_mask_sat = sat_channel <= otsu_thresh_sat

    # Create the Mask (Hue + Saturation channels)
    binary_mask = binary_mask_hue + binary_mask_sat

    # Mask improvements
    # Apply median filtering to remove spurious regions
    #binary_mask = median(binary_mask, np.ones((7, 7)))
    # Dilate to add slight tissue buffer
    binary_mask = binary_dilation(binary_mask, np.ones((5, 5)))

    # Convert to uint8
    binary_mask = (binary_mask + 255).astype(np.uint8)

    # Find contours and create bounding boxes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 750] # Filter small areas
    
    return binary_mask, bounding_boxes 

def is_patch_in_tumor(x, y, patch_size, tumor_annotations):
    for annotation in tumor_annotations.values():
        # Convert annotation to numpy array for easier calculations
        polygon = np.array(annotation)
        
        # Check if any corner of the patch is inside the polygon
        corners = [
            (x, y),
            (x + patch_size, y),
            (x, y + patch_size),
            (x + patch_size, y + patch_size)
        ]
        
        for corner in corners:
            if cv2.pointPolygonTest(polygon, corner, False) >= 0:
                return True
    
    return False

def process_slide(slide_path, annotation_path, model, level=4, patch_size=224):
    # Read the WSI at the specified level
    slide = openslide.OpenSlide(slide_path)
    
    # Extract patches from the tissue region
    binary_mask, bounding_boxes = extract_tissue(slide, level)
    
    # Get the single bounding box encompassing all tissue regions
    xmin = min(box[0] for box in bounding_boxes)
    ymin = min(box[1] for box in bounding_boxes)
    xmax = max(box[0] + box[2] for box in bounding_boxes)
    ymax = max(box[1] + box[3] for box in bounding_boxes)

    # Read tumor annotations if they exist
    tumor_annotations = parse_xml_annotation(annotation_path, level) if annotation_path.exists() else {}
    
    # Print debug information
    print(f"Number of tumor annotations: {len(tumor_annotations)}")
    if tumor_annotations:
        print(f"First tumor annotation: {tumor_annotations[0]}")

    print(f"Slide dimensions at level {level}: {slide.level_dimensions[level]}")
    print(f"Tissue bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")

    results = []
    
    # Extract patches from the single bounding box
    for x in range(xmin, xmax, patch_size):
        for y in range(ymin, ymax, patch_size):
            # Extract and preprocess the patch
            patch = extract_and_preprocess_patch(slide, (x, y), patch_size, level)
            
            # Make prediction
            prediction = predict_patch(model, patch)
            
            # Determine ground truth
            truth = int(is_patch_in_tumor(x, y, patch_size, tumor_annotations))
            
            # Store results
            results.append({
                'slide_path': str(slide_path),
                'tile_loc': f"{x},{y}",
                'prediction': prediction,
                'truth': truth
            })
    print(f"Processed area: ({xmin}, {ymin}) to ({xmax}, {ymax})")
    
    return pd.DataFrame(results)