# Import necessary packages
import numpy as np
import cv2
import openslide
from skimage.filters import threshold_otsu, median
from skimage.morphology import binary_dilation
import xml.etree.ElementTree as ET
import src.utils as utils
import os


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

    factor = 2**mag_level  
    i = 0
    for coords in root.iter('Coordinates'):
        vasc = []
        for coord in coords:
            vasc.append((int(float(coord.attrib.get("X"))/factor),int(float(coord.attrib.get("Y"))/factor)))
        list_annotations[i] = vasc
        i+=1
    return list_annotations

def extract_tissue(rgb_image):

    """
    https://github.com/NMPoole/CS5199-Dissertation/blob/main/src/tools/4_generate_tissue_images.py

    Create a binary mask of the tissue regions using OTSU algorithm in the HSV color space.
    Create bounding boxes around the tissue regions.

    :param rgb_image: Image in an array format 
    :return binary mask: Binary mask of the image (array)
    :return bounding_boxes: list of bounding boxes
    """

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
    binary_mask = median(binary_mask, np.ones((7, 7)))
    # Dilate to add slight tissue buffer
    binary_mask = binary_dilation(binary_mask, np.ones((5, 5)))

    # Convert to uint8
    binary_mask = (binary_mask + 255).astype(np.uint8)

    # Find contours and create bounding boxes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 750] # Filter small areas
    
    return binary_mask, bounding_boxes   

def is_in_tumor_region(patch_x, patch_y, tumor_list):
        
    """
    Check if a given patch defined by its top-left corner (x, y) lies within any of the tumor regions.
    
    :param x: X-coordinate of the top-left corner of the patch.
    :param y: Y-coordinate of the top-left corner of the patch.
    :param tumor_list: List of tumor regions, where each region is defined by a list of coordinates.
    :return: Boolean indicating whether the patch is within any tumor region (True) or not (False).
    """ 
    
    # Calculate the bounding box of the patch.
    patch_size = utils.patch_size
    patch_bbox = [patch_x, patch_y, patch_x + patch_size, patch_y + patch_size]

    # Generate a bounding box for each annotation
    for i in range(len(tumor_list)):
        coords = np.array(tumor_list[i])
        x, y, w, h = cv2.boundingRect(coords)

        # Check if the patch is completely within the bounding rectangle of the tumor region
        if (patch_bbox[0] >= x and patch_bbox[1] >= y # Top-left corner of the patch is inside the tumor region
            and patch_bbox[2] <=y + w and patch_bbox[3] <= y + h):  # Bottom-right corner of the patch is inside the tumor region
            return True

    return False   

def extract_info(path):

    """
    Extract metadata information from a file path.
    
    :param path: The file path from which to extract information.
    :return: A tuple containing the label and ID extracted from the file name.
    """

    # Extract the filename from the path
    filename = os.path.basename(path)
    
    # Split the filename into name and extension
    name, _ = os.path.splitext(filename)
    
    # Split the name into parts
    parts = name.split('_')
    
    # Extract the category and ID
    label = str(parts[0])
    id = str(parts[1])
    
    return label, id