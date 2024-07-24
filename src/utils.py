from pathlib import Path

global_dir = Path("D:/CAMELYON16/data")

#training slides
training_dir = global_dir / "training/"
train_normal_slide_path = training_dir / "normal/"
train_tumor_slide_path = training_dir / "tumor/"
train_annotation_path = training_dir / "lesion_annotations/"

#testing slides
testing_dir = global_dir / "testing/"
test_slide_path = testing_dir / "images/"
test_annotation_path = testing_dir / "lesion_annotations/"

#validation slides
validation_dir = global_dir / "validation/"
val_normal_slide_path = validation_dir / "normal/"
val_tumor_slide_path = validation_dir / "tumor/"
val_annotation_path = validation_dir / "lesion_annotations/"

#patch classification
patch_classification_dir =  global_dir / "patch_classification/"

#training patches
train_patch_dir = patch_classification_dir / "train"
train_normal_patch_path = train_patch_dir / "normal/"
train_tumor_patch_path = train_patch_dir / "tumor/"
train_normal_patch_index = 0
train_tumor_patch_index = 0

#validation patches
val_patch_dir = patch_classification_dir / "val"
val_normal_patch_path = val_patch_dir / "normal/"
val_tumor_patch_path = val_patch_dir / "tumor/"
val_normal_patch_index = 0
val_tumor_patch_index = 0

mag_level = 4
patch_size = 256
nb_patches_per_box = 10

batch_size = 32
nb_epochs = 15
