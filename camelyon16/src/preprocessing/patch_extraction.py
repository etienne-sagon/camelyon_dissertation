#Import necessary packages
import numpy as np
import cv2
import openslide
import sys
import os
from pathlib import Path

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import src.utils as utils
import src.preprocessing.wsi_ops as wsi
import src.preprocessing.data_augmentation as aug

# Example function to extract negative patches from normal WSI
def extract_normal_patches_from_normal_wsi(wsi_path, wsi_image, bounding_boxes, patch_directory, patch_index):
    
    """
    Extract patches from normal regions in a whole slide image (WSI),
    and save them to a directory, including performing data augmentations.
    
    :param wsi_path: Path to the WSI file.
    :param wsi_image: WSI image as a .tif.
    :param bounding_boxes: List of bounding boxes specifying tissue regions in the WSI.
    :param patch_directory: Directory where the extracted patches will be saved.
    :param patch_index: Starting index for naming the saved patches.
    :return: The updated patch index after saving all patches and augmented patches.
    """

    patch_size = utils.patch_size
    patch_per_box = utils.nb_patches_per_box
    level = utils.mag_level_patch
    mag_factor = pow(2, level)

    wsi_label, wsi_id = wsi.extract_info(wsi_path)
    
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        
        for i in range(patch_per_box):
            # Randomly choose patch coordinates within the bounding box
            patch_x = np.random.randint(x, x + w ) #MAYBE ADD "- patch_size"
            patch_y = np.random.randint(y, y + h )
            
            # Read region from WSI and convert to RGB array
            #patch_image = wsi_image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size,:]
            patch_image = wsi_image.read_region((patch_x * mag_factor, patch_y * mag_factor), level, (patch_size, patch_size)).convert("RGB")
            # Convert the PIL image to a NumPy array
            patch_image = np.array(patch_image)

            if (patch_image.shape[0] == patch_size and patch_image.shape[1] == patch_size):
                # Save patch image
                patch_metadata = f"{wsi_label}_ {wsi_id}_{patch_x}_{patch_y}"
                patch_path = f"{patch_directory}/{patch_metadata}_{patch_index}.png"
                cv2.imwrite(patch_path, patch_image)
                patch_index += 1

                #perform data augmentation (rotation, flipping, color jitering)
                patch_index = aug.patch_augmentation(patch_image, patch_directory, patch_metadata, patch_index)       
                             
    return patch_index


def extract_normal_patches_from_tumor_wsi(wsi_path, wsi_image, anno_path, bounding_boxes, patch_directory, patch_index):
    
    """
    Extract patches from non-tumor regions in a whole slide image (WSI) of a tumor,
    and save them to a directory, including performing data augmentations.
    
    :param wsi_path: Path to the WSI file.
    :param wsi_image: WSI image as a .tif.
    :param anno_path: Path to the annotation file (XML) containing tumor regions.
    :param bounding_boxes: List of bounding boxes specifying tissue regions in the WSI.
    :param patch_directory: Directory where the extracted patches will be saved.
    :param patch_index: Starting index for naming the saved patches.
    :return: The updated patch index after saving all patches and augmented patches.
    """
    
    patch_size = utils.patch_size
    patch_per_box = utils.nb_patches_per_box
    level = utils.mag_level_patch
    mag_factor = pow(2, level)
    
    # Extract the annotations from xml file (scaled for the magnification level used)
    annolist = wsi.parse_xml_annotation(anno_path, level)

    wsi_label, wsi_id = wsi.extract_info(wsi_path)

    for bbox in bounding_boxes:
        x, y, w, h = bbox
        
        patches_extracted = 0
        while patches_extracted < patch_per_box:
            # Randomly choose patch coordinates within the bounding box#
            patch_x = np.random.randint(x, x + w)
            patch_y = np.random.randint(y, y + h)
            
            # Check if the patch is in the tumor region
            if not wsi.is_in_tumor_region(patch_x, patch_y, annolist):

                # Read region from WSI and convert to array
                #patch_image = wsi_image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size, :]
                patch_image = wsi_image.read_region((patch_x * mag_factor, patch_y * mag_factor), level, (patch_size, patch_size)).convert("RGB")
                patch_image = np.array(patch_image)


                if (patch_image.shape[0] == patch_size and patch_image.shape[1] == patch_size):
                    # Save patch image
                    patch_metadata = f"{wsi_label}_ {wsi_id}_{patch_x}_{patch_y}"
                    patch_path = f"{patch_directory}/{patch_metadata}_{patch_index}.png"
                    cv2.imwrite(patch_path, patch_image)
                    patch_index += 1
                
                    #perform data augmentation (rotation, flipping, color jitering)
                    patch_index = aug.patch_augmentation(patch_image, patch_directory, patch_metadata, patch_index)
                patches_extracted += 1

    return patch_index


def extract_tumor_patches_from_tumor_wsi(wsi_path, wsi_image, anno_path, patch_directory, patch_index):

    """
    Extract patches from tumor regions in a whole slide image (WSI),
    and save them to a directory, including performing data augmentations.
    
    :param wsi_path: Path to the WSI file.
    :param wsi_image: WSI image as .tif.
    :param anno_path: Path to the annotation file (XML) containing tumor regions.
    :param patch_directory: Directory where the extracted patches will be saved.
    :param patch_index: Starting index for naming the saved patches.
    :return: The updated patch index after saving all patches and augmented patches.
    """

    patch_size = utils.patch_size
    patch_per_box = utils.nb_patches_per_box
    level = utils.mag_level_patch
    mag_factor = pow(2, level)

    #extract the annotations from xml file (already scaled for the magnification level used)
    annolist = wsi.parse_xml_annotation(anno_path, level)

    wsi_label, wsi_id = wsi.extract_info(wsi_path)
    
    #generate bounding boxes for each annotation
    for i in range(len(annolist)):                   
        coords = np.array(annolist[i])
        x, y, w, h = cv2.boundingRect(coords)

        #for each bounding box
        for j in range(patch_per_box):  

            # we sample random coordinates
            patch_x = np.random.randint(x, x + w) #MAYBE ADD "- patch_size"
            patch_y = np.random.randint(y, y + h)                      
            offsetx = np.random.randint(-50, 50)
            offsety = np.random.randint(-50, 50)
            spointx, spointy = patch_x + offsetx, patch_y + offsety 
            
            #we extract a patch and save it to the right directory
            #patch_image = wsi_image[spointy:spointy + patch_size, spointx:spointx + patch_size,:]
            patch_image = wsi_image.read_region((spointx * mag_factor, spointy * mag_factor), level, (patch_size, patch_size)).convert("RGB")
            patch_image = np.array(patch_image)

            if (patch_image.shape[0] == patch_size and patch_image.shape[1] == patch_size):

                patch_metadata = f"{wsi_label}_ {wsi_id}_{spointx}_{spointy}"
                patch_path = f"{patch_directory}/{patch_metadata}_{patch_index}.png"
                cv2.imwrite(patch_path, patch_image)
                patch_index += 1

                #perform data augmentation (rotation, flipping, color jitering)
                patch_index = aug.patch_augmentation(patch_image, patch_directory, patch_metadata, patch_index)

    return patch_index

def extract_save_patches_training():

    patch_normal_index = utils.train_normal_patch_index
    patch_tumor_index = utils.train_tumor_patch_index
    train_normal_slide_path = Path(utils.train_normal_slide_path)
    train_tumor_slide_path = Path(utils.train_tumor_slide_path)

    print("\n\nPatch extraction is starting...")

    print("\n\nExtracting normal patches from normal WSI...")
    print(train_normal_slide_path)

    # Check if the directory exists
    if not train_normal_slide_path.exists():
        print(f"Error: Directory {train_normal_slide_path} does not exist.")
        return

    
    # List all .tif files in the directory
    tif_files = list(train_normal_slide_path.glob("*.tif"))
    # Print the number of .tif files
    print(f"Total number of .tif files in '{train_normal_slide_path}': {len(tif_files)}")

    # NORMAL PATCH FROM NORMAL SLIDES
    for path in train_normal_slide_path.glob("*.tif"):
        print(path)
        #read the wsi
        slide = openslide.OpenSlide(path)
        #extract the tissue
        _, bounding_boxes = wsi.extract_tissue(slide)
        
        temp_normal = patch_normal_index

        #extract normal patches 
        patch_normal_index = extract_normal_patches_from_normal_wsi(path, slide, bounding_boxes, utils.train_normal_patch_path, patch_normal_index)
        slide.close()
        print(f"FOR : {path} \nNumber of patch extracted : {patch_normal_index - temp_normal} to {utils.train_normal_patch_path}" )

    print("\n\nExtracting normal and tumor patches from tumor WSI...")
    print(train_tumor_slide_path)  

    # Check if the directory exists
    if not train_tumor_slide_path.exists():
        print(f"Error: Directory {train_tumor_slide_path} does not exist.")
        return
    
    # NORMAL PATCH FROM TUMOR SLIDEs
    for path in train_tumor_slide_path.glob("*.tif"):
        print(path)
        #read the wsi
        slide = openslide.OpenSlide(path)
        _, wsi_id = wsi.extract_info(str(path))

        #extract the tissue
        _, bounding_boxes = wsi.extract_tissue(slide)
        #read the annotations
        xml_path = utils.train_annotation_path / f"tumor_{int(wsi_id):03d}.xml"

        temp_normal = patch_normal_index
        #extract normal patches 
        patch_normal_index = extract_normal_patches_from_tumor_wsi(path, slide, xml_path, bounding_boxes, utils.train_normal_patch_path, patch_normal_index)
        print(f"FOR : {path}\nNumber of normal patch extracted : {patch_normal_index - temp_normal} to {utils.train_normal_patch_path}")

        #extract tumor patches 
        temp_tumor = patch_tumor_index
        patch_tumor_index = extract_tumor_patches_from_tumor_wsi(path,  slide, xml_path, utils.train_tumor_patch_path, patch_tumor_index)
        print(f"\nNumber of tumor patch extracted : {patch_tumor_index - temp_tumor} to {utils.train_tumor_patch_path}")
        
        slide.close()

    
    print("\n\nTOTAL NORMAL PATCHES EXTRACTED : ",patch_normal_index)
    print("\nTOTAL TUMOR PATCHES EXTRACTED : ",patch_tumor_index)