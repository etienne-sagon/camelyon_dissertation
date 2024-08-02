import src.preprocessing.patch_extraction as patch_extract
import src.utils as utils
import src.training_models.googlenet_train as train
import os

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
    
    # Extract and save the packages from the WSIs
    #patch_extract.extract_save_patches_training()

    train.train_googlenet_model()

if __name__ == "__main__":
    main()
