import src.preprocessing.patch_extraction as patch_extract
import src.utils as utils

def main():
    # Extract and save the packages from the WSIs
    patch_extract.extract_save_patches_training()

if __name__ == "__main__":
    main()