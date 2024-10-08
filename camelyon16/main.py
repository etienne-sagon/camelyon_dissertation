import src.preprocessing.patch_extraction as patch_extract
import src.utils as utils
import src.training_models.googlenet_train as train
import src.inference.patch_inference_test as patch_inf
import src.inference.patch_inference_val as patch_inf_val

import os
from pathlib import Path
import pandas as pd
import time


def check_directory(path):
    if not os.path.exists(path):
        print(f"Warning: Directory {path} does not exist.")
    else:
        print(f"Directory {path} exists.")
        print("Contents:")
        print(os.listdir(path))

def main():
    print(f"Patch mag level: {utils.mag_level_patch}")
    print(f"Tissue mag level: {utils.mag_level_tissue}")
    print(f"Patch size: {utils.patch_size}")
    print(f"Patches per box: {utils.nb_patches_per_box}")
    print(f"Current working directory: {os.getcwd()}")

    print("Contents of /app/data directory:")
    try:
        print(os.listdir("/app/data"))
    except FileNotFoundError:
        print("The /app/data directory does not exist.")

    print("Contents of /app directory:")
    print(os.listdir("/app"))

    # check_directory("/app/data/testing/images")
    # check_directory("/app/data/testing/lesion_annotations")

    print("Starting...")
    
    # Extract and save the packages from the WSIs
    # patch_extract.extract_save_patches_training()
    
    # train.train_googlenet_model()
    # print("batch size = ", utils.batch_size)
    # print("nb epochs = ", utils.nb_epochs)
   

    model_path = utils.model_save_path
    # model_path = "D:\CAMELYON16\camelyon_dissertation\models\googlenet_test.pth"
    model = patch_inf.load_model(model_path)

    # # Define paths
    # x_path = Path("D:/CAMELYON16/data/testing/camelyonpatch_level_2_split_test_x.h5")
    # y_path = Path("D:/CAMELYON16/data/testing/camelyonpatch_level_2_split_test_y.h5")

    # x_path = Path("/app/scripts/patch_data/camelyonpatch_level_2_split_test_x.h5")
    # y_path = Path("/app/scripts/patch_data/camelyonpatch_level_2_split_test_y.h5")
    # metadata_path = Path("/app/scripts/patch_data/camelyonpatch_level_2_split_test_meta.csv")

    inference_start_time = time.time()

    # Load and preprocess the data
    dataloader = patch_inf_val.load_and_preprocess_validation_data(utils.val_normal_patch_path, utils.val_tumor_patch_path, 32)
    # dataloader = patch_inf.load_and_preprocess_data(x_path, y_path, 32)

    final_df = patch_inf_val.predict_patches(model, dataloader)

    print(final_df)

    # output_dir = Path("D:/CAMELYON16/camelyon_dissertation/test_metrics")

    patch_inf_val.compute_and_save_metrics(final_df, utils.inference_save_path_val)

    # #patch_inf.create_heatmap(wsi_path, metadata_path, final_df, output_path, level=5)

    inference_end_time = time.time()
    
    inference_time = inference_end_time - inference_start_time #duration of training
    print("Total inference time: ",inference_time)

    # Move patches before loading data
    # train.move_patches_to_validation(utils.train_normal_patch_path, utils.val_normal_patch_path, 0.15)
    # train.move_patches_to_validation(utils.train_tumor_patch_path, utils.val_tumor_patch_path, 0.15)


if __name__ == "__main__":
    main()
