import src.preprocessing.patch_extraction as patch_extract
import src.utils as utils
import src.training_models.googlenet_train as train

def main():
    print(utils.mag_level, utils.patch_size, utils.nb_patches_per_box)
    # Extract and save the packages from the WSIs
    patch_extract.extract_save_patches_training()

    train.train_googlenet_model()

if __name__ == "__main__":
    main()
