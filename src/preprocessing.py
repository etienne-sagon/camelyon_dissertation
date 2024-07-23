#Import necessary packages
import numpy as np
import cv2

import src.utils as utils
import src.wsi_ops as wsi
import src.data_augmentation as aug

# Example function to extract negative patches from normal WSI
def extract_normal_patches_from_normal_wsi(wsi_path, wsi_image, bounding_boxes, patch_directory, patch_index):
    
    """
    Extract patches from normal regions in a whole slide image (WSI),
    and save them to a directory, including performing data augmentations.
    
    :param wsi_path: Path to the WSI file.
    :param wsi_image: WSI image as a NumPy array.
    :param bounding_boxes: List of bounding boxes specifying tissue regions in the WSI.
    :param patch_directory: Directory where the extracted patches will be saved.
    :param patch_index: Starting index for naming the saved patches.
    :return: The updated patch index after saving all patches and augmented patches.
    """

    patch_size = utils.patch_size
    patch_per_box = utils.nb_patches_per_box
    mag_factor = pow(2, utils.mag_level)

    wsi_label, wsi_id = wsi.extract_info(wsi_path)
    
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        
        for i in range(patch_per_box):
            # Randomly choose patch coordinates within the bounding box
            patch_x = np.random.randint(x, x + w ) #MAYBE ADD "- patch_size"
            patch_y = np.random.randint(y, y + h )
            
            # Read region from WSI and convert to RGB array
            patch_image = wsi_image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size,:]

            if (patch_image.shape[0] == 299 and patch_image.shape[1] == 299):
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
    :param wsi_image: WSI image as a NumPy array.
    :param anno_path: Path to the annotation file (XML) containing tumor regions.
    :param bounding_boxes: List of bounding boxes specifying tissue regions in the WSI.
    :param patch_directory: Directory where the extracted patches will be saved.
    :param patch_index: Starting index for naming the saved patches.
    :return: The updated patch index after saving all patches and augmented patches.
    """
    
    patch_size = utils.patch_size
    patch_per_box = utils.nb_patches_per_box
    mag_factor = pow(2,utils.mag_level)
    
    # Extract the annotations from xml file (scaled for the magnification level used)
    annolist = wsi.parse_xml_annotation(anno_path, utils.mag_level)

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
                patch_image = wsi_image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size, :]

                if (patch_image.shape[0] == 299 and patch_image.shape[1] == 299):
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
    https://thedeeplearning-intern322.medium.com/extracting-patches-from-whole-slide-images-wsi-c3bb9b00d9b5

    Extract patches from tumor regions in a whole slide image (WSI),
    and save them to a directory, including performing data augmentations.
    
    :param wsi_path: Path to the WSI file.
    :param wsi_image: WSI image as a NumPy array.
    :param anno_path: Path to the annotation file (XML) containing tumor regions.
    :param patch_directory: Directory where the extracted patches will be saved.
    :param patch_index: Starting index for naming the saved patches.
    :return: The updated patch index after saving all patches and augmented patches.
    """

    patch_size = utils.patch_size
    patch_per_box = utils.nb_patches_per_box
    mag_factor = pow(2,utils.mag_level)

    #extract the annotations from xml file (already scaled for the magnification level used)
    annolist = wsi.parse_xml_annotation(anno_path, utils.mag_level)

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
            patch_image = wsi_image[spointy:spointy + patch_size, spointx:spointx + patch_size,:]
            if (patch_image.shape[0] == 299 and patch_image.shape[1] == 299):

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
    print("Patch extraction is starting...")

    print("Extracting normal patches from normal WSI...")
    # NORMAL PATCH FROM NORMAL SLIDES
    for path in utils.train_normal_slide_path.glob("*.png"):
        #read the wsi
        wsi_image = cv2.imread(str(path))
        #extract the tissue
        _, bounding_boxes = wsi.extract_tissue(wsi_image)
        
        temp_normal = patch_normal_index

        #extract normal patches 
        patch_normal_index = extract_normal_patches_from_normal_wsi(path, wsi_image, bounding_boxes, utils.train_normal_patch_path, patch_normal_index)
        
        print(f"FOR : {path} \nNumber of patch extracted : {patch_normal_index - temp_normal}")

    print("Extracting normal patches from tumor WSI...")
    # NORMAL PATCH FROM TUMOR SLIDEs
    for path in utils.train_tumor_slide_path.glob("*.png"):
        #read the wsi
        wsi_image = cv2.imread(str(path))
        _, wsi_id = wsi.extract_info(str(path))

        #extract the tissue
        _, bounding_boxes = wsi.extract_tissue(wsi_image)
        #read the annotations
        xml_path = utils.train_annotation_path / f"tumor_{int(wsi_id):03d}.xml"

        temp_normal = patch_normal_index
        #extract normal patches 
        patch_normal_index = extract_normal_patches_from_tumor_wsi(path, wsi_image, xml_path, bounding_boxes, utils.train_normal_patch_path, patch_normal_index)
        print(f"FOR : {path}\nNumber of patch extracted : {patch_normal_index - temp_normal}")
    

    print("Extracting tumor patches from tumor WSI...")
    # TUMOR PATCH FROM TUMOR SLIDE (TRAINING)
    for path in utils.train_tumor_slide_path.glob("*.png"):
        #read the wsi
        wsi_image = cv2.imread(str(path))
        _, wsi_id = wsi.extract_info(str(path))

        #read the annotations
        xml_path = utils.train_annotation_path / f"tumor_{int(wsi_id):03d}.xml"

        temp_tumor = patch_tumor_index

        #extract tumor patches 
        patch_tumor_index = extract_tumor_patches_from_tumor_wsi(path,  wsi_image, xml_path, utils.train_tumor_patch_path, patch_tumor_index)
        print(f"FOR : {path} \nNumber of patch extracted : {patch_tumor_index - temp_tumor}")
        
    
    print("\nTOTAL NORMAL PATCHES EXTRACTED : ",patch_normal_index)
    print("\nTOTAL TUMOR PATCHES EXTRACTED : ",patch_tumor_index)