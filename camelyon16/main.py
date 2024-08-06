import src.preprocessing.patch_extraction as patch_extract
import src.utils as utils
import src.training_models.googlenet_train as train
import src.inference.inference_googlenet as inference
import src.inference.patch_inference as patch_inf
import os
from pathlib import Path
import pandas as pd
import openslide

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

    # print("Contents of /app/data directory:")
    # try:
    #     print(os.listdir("/app/data"))
    # except FileNotFoundError:
    #     print("The /app/data directory does not exist.")

    # print("Contents of /app directory:")
    # print(os.listdir("/app"))

    # check_directory("/app/data")
    # check_directory("/app/data/training")
    # check_directory("/app/data/training/normal")
    # check_directory("/app/data/training/tumor")

    check_directory("/app/data/testing/images")
    check_directory("/app/data/testing/lesion_annotations")

    # Extract and save the packages from the WSIs
    #patch_extract.extract_save_patches_training()

    #train.train_googlenet_model()

    print("Starting...")

    model_path = utils.model_save_path
    #model_path = "D:\CAMELYON16\camelyon_dissertation\models\googlenet_test.pth"
    model = patch_inf.load_model(model_path)

    # test_slide_example = Path("D:/CAMELYON16/data/testing/images")
    # test_anno_example = Path("D:/CAMELYON16/data/testing/lesion_annotations")
    
    all_results = []
    image_count = 0

    for slide_file in utils.test_slide_path.glob('*.tif'):
        
        if image_count >= 5:
            break

        slide_path = slide_file
        annotation_path = utils.test_annotation_path / slide_file.with_suffix('.xml').name

        print(slide_path)
        print(annotation_path)
        level = 5

        results = patch_inf.process_slide(slide_path, annotation_path, model, level)
        all_results.append(results)

        image_count += 1
        print(f"image count : {image_count}")
    
    # Combine all results into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)

    print(final_df.shape)
    print("\n",final_df)
    # Print overall ground truth counts using value_counts()
    print("\nOverall Ground Truth Counts:")
    truth_counts = final_df['truth'].value_counts().sort_index()
    print(truth_counts)
    print(f"Total patches: {truth_counts.sum()}")

    patch_inf.compute_and_save_metrics(final_df, utils.inference_save_path)


if __name__ == "__main__":
    main()
