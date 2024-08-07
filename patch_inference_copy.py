import openslide
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from torchvision import transforms
import xml.etree.ElementTree as ET
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score

import src.preprocessing.wsi_ops as ops
import src.utils as utils

def create_tumor_mask_V1(slide_dimensions, xml_path, level):
    print("Masking...")
    width, height = slide_dimensions
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mag_factor = pow(2,level)

    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extract coordinates from the XML
    polygons = []
    for annotation in root.findall('.//Annotation'):
        coords = annotation.find('Coordinates')
        points = [(int(float(coord.attrib.get("X"))/mag_factor), int(float(coord.attrib.get("Y"))/mag_factor)) for coord in coords.findall('Coordinate')]
        polygons.append(Polygon(points))
    print("Coords extracted")

    # Create the mask
    for y in range(height):
        for x in range(width):
            point = Point(x, y)
            if any(polygon.contains(point) for polygon in polygons):
                mask[y, x] = 255
    print("Binary mask created")

    return mask

def create_tumor_mask(slide_dimensions, xml_path, level):
    print("Masking...")
    width, height = slide_dimensions
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mag_factor = pow(2, level)

    # Extract coordinates and create polygons
    polygons = []
    for annotation in root.findall('.//Annotation'):
        coords = annotation.find('Coordinates')
        points = [(int(float(coord.attrib.get("X"))/mag_factor), int(float(coord.attrib.get("Y"))/mag_factor)) 
                  for coord in coords.findall('Coordinate')]
        if len(points) >= 3:  # Ensure we have at least 3 points to make a polygon
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Attempt to fix invalid polygons
            if poly.is_valid:
                polygons.append(poly)
    print("Coords extracted")

    # Combine polygons
    try:
        combined_polygon = unary_union(polygons)
    except Exception as e:
        print(f"Error in unary_union: {e}")
        combined_polygon = MultiPolygon(polygons)  # Fallback to MultiPolygon if union fails

    # Create a PIL image for drawing
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # Draw the polygons
    if isinstance(combined_polygon, (Polygon, MultiPolygon)):
        if isinstance(combined_polygon, Polygon):
            polygons_to_draw = [combined_polygon]
        else:
            polygons_to_draw = list(combined_polygon.geoms)
        
        for polygon in polygons_to_draw:
            if polygon.is_valid:
                exterior_coords = list(polygon.exterior.coords)
                draw.polygon(exterior_coords, outline=255, fill=255)
                for interior in polygon.interiors:
                    interior_coords = list(interior.coords)
                    draw.polygon(interior_coords, outline=0, fill=0)

    print("Binary mask created")

    output_path = f"D:/CAMELYON16/camelyon_dissertation/camelyon16/mask.png"
    inf_save_path = Path(utils.inference_save_path )
    file_name = xml_path.stem
    output_path = inf_save_path / f"tumour_mask_{file_name}.png"
    # Convert PIL image to numpy array
    mask = np.array(mask_img)
    save_mask_as_image(mask,output_path)
    
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
    print("GoogLeNet loaded")

    return model

# Function to make prediction for a single patch
def predict_patch(model, patch):
    with torch.no_grad():
        output = model(patch)
        probabilities = torch.softmax(output, dim=1)
    return probabilities[0, 1].item()


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


def compute_and_save_metrics(df, output_dir):

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert predictions to binary (assuming threshold of 0.5)
    y_pred_proba = df['prediction']
    y_true = df['truth']


    # Compute ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)


    # Compute Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)

    # # Find optimal threshold using F1 score
    # f1_scores = []
    # for threshold in pr_thresholds:
    #     y_pred = (y_pred_proba >= threshold).astype(int)
    #     f1_scores.append(f1_score(y_true, y_pred))
    # optimal_threshold = pr_thresholds[np.argmax(f1_scores)]

    optimal_threshold = 0.5
    # Compute metrics using optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision_sc = precision_score(y_true, y_pred)
    recall_sc = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Save metrics to a text file
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"\nOptimal Threshold: {optimal_threshold:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision_sc:.4f}\n")
        f.write(f"Recall: {recall_sc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {roc_auc:.4f}\n")
        f.write(f"Average Precision: {average_precision:.4f}\n")

    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()

    print(f"Metrics and ROC curve saved in {output_dir}")

    # Plot and save Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(output_dir / 'precision_recall_curve.png')
    plt.close()
    
    print(f"Metrics and curves saved in {output_dir}")


def process_slide(slide_path, annotation_path, model, level, img_nb, patch_size=224):
    # Read the WSI at the specified level
    slide = openslide.OpenSlide(slide_path)
    slide_dim = slide.level_dimensions[level]
    print(f"Processing slide: {slide_path}")
    print(f"Annotation path: {annotation_path}")
    print(f"Slide dimensions at level {level}: {slide_dim}")

    # Extract patches from the tissue region
    _, bounding_boxes = ops.extract_tissue(slide, level)
    
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

    if img_nb <= 3:
        # Visualize the tumor mask and extracted patches
        inference_save_path = Path(utils.inference_save_path)
        wsi_name = slide_path.stem
        vis_save_path = inference_save_path / f"viz_{wsi_name}.png"
        #vis_save_path = "D:/CAMELYON16/camelyon_dissertation/processing_visualization.png"
        if tumor_mask is not None:
            visualize_slide_processing(slide_path, level, tumor_mask, extracted_patches, vis_save_path)
            print(f"Processing visualization saved to: {vis_save_path}")
    
    # Assuming your DataFrame is named 'final_df'
    results_df = pd.DataFrame(results)
    return results_df