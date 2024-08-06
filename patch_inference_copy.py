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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
from PIL import Image

import src.preprocessing.wsi_ops as ops

def create_tumor_mask(slide_dimensions, xml_path, level):
    width, height = slide_dimensions
    print(f"Input slide dimensions at level {level}: ({width}, {height})")

    print("masking...")
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mag_factor = pow(2,level)

    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    print("empty mask created")

    # Extract coordinates from the XML
    polygons = []
    for annotation in root.findall('.//Annotation'):
        coords = annotation.find('Coordinates')
        points = [(int(float(coord.attrib.get("X"))/mag_factor), int(float(coord.attrib.get("Y"))/mag_factor)) for coord in coords.findall('Coordinate')]
        polygons.append(Polygon(points))
    print("coords extracted")

    # Create the mask
    for y in range(height):
        print(y)
        for x in range(width):
            point = Point(x, y)
            if any(polygon.contains(point) for polygon in polygons):
                mask[y, x] = 255
    print("binary mask created")

    return mask

def save_mask_as_image(mask, output_path):
    """
    Save the binary mask as an image file
    """
    img = Image.fromarray(mask)
    img.save(output_path)

# Function to extract and preprocess a patch
def extract_and_preprocess_patch(slide, coords, level, patch_size=224):
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


def visualize_slide_processing(slide_path, level, tumor_mask, extracted_patches, save_path):
    # Open the slide
    slide = openslide.OpenSlide(str(slide_path))
    
    # Read the whole slide image at the specified level
    wsi_img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')
    wsi_img = np.array(wsi_img)

    # Ensure tumor_mask has the same dimensions as wsi_img
    tumor_mask = np.resize(tumor_mask, (wsi_img.shape[0], wsi_img.shape[1]))

    # Create a color mask (red for tumor)
    color_mask = np.zeros((*tumor_mask.shape, 4), dtype=np.uint8)
    color_mask[tumor_mask > 0] = [255, 0, 0, 128]  # Red with 50% opacity


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
    
    # Plot tumor mask
    ax1.imshow(tumor_mask, cmap='binary')
    ax1.set_title(f'Tumor Mask')
    ax1.axis('off')
    
    # Plot WSI with overlaid tumor mask and extracted patches
    ax2.imshow(wsi_img)
    ax2.imshow(color_mask)
    for patch in extracted_patches:
        rect = patches.Rectangle((patch[0], patch[1]), patch[2], patch[3], 
                                 linewidth=1, edgecolor='blue', facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title('WSI with Tumor Mask and Extracted Patches')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Close the slide
    slide.close()



def process_slide(slide_path, annotation_path, model, level, patch_size=224):
    # Read the WSI at the specified level
    slide = openslide.OpenSlide(slide_path)
    slide_dim = slide.level_dimensions[level]
    print(f"Processing slide: {slide_path}")
    print(f"Annotation path: {annotation_path}")
    print(f"Slide dimensions at level {level}: {slide_dim}")

    
    # Extract patches from the tissue region
    tissue_mask, bounding_boxes = extract_tissue(slide, level)
    
    # Get the single bounding box encompassing all tissue regions
    xmin = min(box[0] for box in bounding_boxes)
    ymin = min(box[1] for box in bounding_boxes)
    xmax = max(box[0] + box[2] for box in bounding_boxes)
    ymax = max(box[1] + box[3] for box in bounding_boxes)

    # Read tumor annotations if they exist
    if annotation_path.exists():
        tumor_mask = create_tumor_mask(slide_dim, annotation_path, level)
    else:
        tumor_mask = None
    
    # Print debug information
    #print(f"Number of tumor annotations: {len(tumor_annotations)}")

    print(f"Slide dimensions at level {level}: {slide.level_dimensions[level]}")
    print(f"Tissue bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")

    results = []
    extracted_patches = []
    
    # Extract patches from the single bounding box
    for x in range(xmin, xmax, patch_size):
        for y in range(ymin, ymax, patch_size):
            # Extract and preprocess the patch
            patch = extract_and_preprocess_patch(slide, (x, y), level, patch_size)
            
            # Make prediction
            prediction = predict_patch(model, patch)
            
            # Determine ground truth from the tumor mask
            if tumor_mask is not None:
                truth = int(np.any(tumor_mask[y:y+patch_size, x:x+patch_size]))
            else:
                truth = 0
            
            # Store results
            results.append({
                'slide_path': str(slide_path),
                'tile_loc': f"{x},{y}",
                'prediction': prediction,
                'truth': truth
            })

            extracted_patches.append((x, y, patch_size, patch_size))

    # Visualize the tumor mask and extracted patches
    vis_save_path = Path("D:/CAMELYON16/camelyon_dissertation/processing_visualization.png")
    if tumor_mask is not None:
        visualize_slide_processing(slide_path, level, tumor_mask, extracted_patches, vis_save_path)
        print(f"Processing visualization saved to: {vis_save_path}")
    
    return pd.DataFrame(results)