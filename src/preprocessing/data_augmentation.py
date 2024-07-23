#import necessary packages
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2


def rotate_90(patch_image):

    """
    Rotate a given image patch by a random multiple of 90 degrees counter-clockwise.
    
    :param patch_image: A NumPy array representing the image patch.
    :return: The rotated image patch in an array format
    """
    
    #ADD 0 WHEN INTEGRATED TO PIPELINE -> ?????????????????
    angles = [1, 2, 3]  # Corresponds to 90, 180, 270 degree rotations counter-clockwise.
    angle = np.random.choice(angles)

    rotated_patch_img = np.rot90(patch_image, k=angle)
 
    return rotated_patch_img

def random_flip(patch_image):

    """
    Randomly flip a given image patch either horizontally, vertically, or both.
    
    :param patch_image: A NumPy array representing the image patch.
    :return: The flipped image patch in an array format
    """
    # Randomly decide whether to flip horizontally and/or vertically
    flip_horizontal, flip_vertical = False, False

    while (flip_horizontal == False and flip_vertical == False):
        
        flip_horizontal = np.random.choice([True, False])
        flip_vertical = np.random.choice([True, False])

    # Perform flips if chosen
    if flip_horizontal:
        flipped_patch_img = np.flip(patch_image, axis=1)
    if flip_vertical:
        flipped_patch_img = np.flip(patch_image, axis=0)

    return flipped_patch_img

def color_jitter(patch_array):
     
    """"
    Apply color jittering to a given image patch.
    Adjust its brightness, contrast, saturation, and hue.
    
    :param patch_array: A NumPy array representing the image patch.
    :return: The color-jittered image patch as a NumPy array.
    """
    
    # Define the color jittering transform
    color_jitter = transforms.ColorJitter(
        brightness=0.3,  # Adjust brightness
        contrast=0.3,    # Adjust contrast
        saturation=0.3,  # Adjust saturation
        hue=0.15          # Adjust hue
    )

    # Convert NumPy array to PIL Image
    patch_image = Image.fromarray(patch_array)

    # Apply the color jittering transform
    augmented_patch_image = color_jitter(patch_image)

    # Convert the augmented image back to a NumPy array
    augmented_patch_array = np.array(augmented_patch_image)

    return augmented_patch_array


def patch_augmentation(patch_image, patch_directory, patch_metadata, patch_index):

    """
    Apply various augmentations to a given image patch 
    and save the augmented patches to a specified directory.
    
    :param patch_image: A NumPy array representing the image patch.
    :param patch_directory: Directory where the augmented patches will be saved.
    :param patch_metadata: Metadata to include in the filenames of the saved patches (wsi_label, wsi_id, x and y coords).
    :param patch_index: Starting index for naming the saved patches.
    :return: The updated patch index after saving all augmented patches.
    """

    augmented_patches = []

    # Create augmented patches qnd add them to the list
    aug_patch = rotate_90(patch_image)
    augmented_patches.append(aug_patch)
    augmented_patches.append(random_flip(patch_image))
    augmented_patches.append(color_jitter(patch_image))

    # Save each augmented patches to the specified directory
    for augmented_patch in augmented_patches:
        patch_path = f"{patch_directory}/{patch_metadata}_{patch_index}.png"
        cv2.imwrite(patch_path, augmented_patch)
        patch_index += 1

    return patch_index