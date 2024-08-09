import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import h5py
from tqdm import tqdm
import openslide

def get_googlenet_model(num_classes=2):

    """
    Create and return a GoogLeNet model with a custom number of output classes.

    :param num_classes: The number of output classes for the model's final fully connected layer. Defaults to 2, suitable for binary classification.
    :return: A GoogLeNet model instance with its final fully connected layer modified to output the specified number of classes.
    """

    model = models.googlenet(weights=None, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# Load your GoogLeNet model
def load_model(model_path):
    """
    Load a GoogLeNet model with a custom architecture from a saved state dictionary.

    :param model_path: Path to the saved model state dictionary (a .pth file).
    
    :return: The GoogLeNet model loaded with the specified weights, ready for inference.
    """

    DEVICE = torch.device("cpu")
    
    # Initialize the model architecture
    model = get_googlenet_model(num_classes=2, pretrained=False)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # If the state_dict was saved with DataParallel, remove the 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    

    # Load the state dict into the model   
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    print("GoogLeNet loaded")

    return model


# Function to load and preprocess data
def load_and_preprocess_data(x_path, y_path, batch_size=32):
    """
    Load image data and labels from HDF5 files, preprocess the images, and create a DataLoader for batch-wise data loading.

    :param x_path: Path to the HDF5 file containing the image data.
    :param y_path: Path to the HDF5 file containing the labels.
    :param batch_size: The number of samples per batch in the DataLoader. Defaults to 32.
    
    :return: A PyTorch DataLoader object containing the preprocessed image tensors and corresponding labels.
    """

    with h5py.File(x_path, 'r') as hf_x, h5py.File(y_path, 'r') as hf_y:
        x = hf_x['x'][:]
        y = hf_y['y'][:]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = []
    for img, label in zip(x, y):
        img_tensor = transform(img)
        dataset.append((img_tensor, label[0]))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def predict_patches(model, dataloader):
    """
    Use a trained model to make predictions on a dataset of image patches and return the results in a DataFrame.

    :param model: The trained PyTorch model used for making predictions.
    :param dataloader: A DataLoader object containing the preprocessed image patches and their corresponding labels.
    
    :return: A pandas DataFrame containing the true labels and predicted probabilities of the positive class for each image patch.
    """

    # Make predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch, labels in tqdm(dataloader, desc="Making predictions"):
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            predictions.extend(probs.cpu().numpy())
            
            # Extract scalar values from labels
            labels = labels.cpu().numpy()
            true_labels.extend(labels.flatten())  # Flatten to ensure 1D array
    
    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'truth': true_labels,
        'prediction': predictions
    })

    return results_df

def compute_and_save_metrics(df, output_dir):
    """
    Compute various classification metrics and save the results, to the specified output directory.

    :param df: A pandas DataFrame containing the true labels and predicted probabilities. It should have two columns: 'truth' (true labels) and 'prediction' (predicted probabilities).
    :param output_dir: Directory where the metrics and plots will be saved.
    
    :return: None
    """


    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert predictions to binary (assuming threshold of 0.5)
    y_pred_proba = df['prediction']
    y_true = df['truth']

    # Compute ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)

    # Compute metrics for different thresholds
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    recalls=[]
    precisions = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    

    # Plot precision, accuracy, and F1 score for different thresholds
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Accuracy, and F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'metrics_vs_threshold.png')
    plt.close()

    

    optimal_threshold = 0.94
    # Compute metrics using optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision_sc = precision_score(y_true, y_pred)
    recall_sc = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

   # Save metrics to a text file
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"\nOptimal Threshold: {optimal_threshold:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision_sc:.4f}\n")
        f.write(f"Recall: {recall_sc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {roc_auc:.4f}\n")
        f.write(f"Average Precision: {average_precision:.4f}\n")

    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()

    # Plot and save Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(output_dir / 'precision_recall_curve.png')
    plt.close()
    
    print(f"Metrics and curves saved in {output_dir}")

def create_heatmap(wsi_path, metadata_path, predictions_df, output_path, level=2, patch_size=96):
    """
    Generate and save a tumor probability heatmap overlaid on a whole-slide image (WSI).

    :param wsi_path: Path to the whole-slide image (WSI) file.
    :param metadata_path: Path to the CSV file containing metadata with coordinates for patches.
    :param predictions_df: A DataFrame containing tumor predictions for each patch.
    :param output_path: Path where the resulting heatmap image will be saved.
    :param level: The resolution level of the WSI to use for the heatmap. Defaults to 2.
    :param patch_size: The size of each patch in the original WSI resolution. Defaults to 96.
    
    :return: None. The heatmap is saved as an image to the specified output path.
    """

    # Read the WSI
    wsi = openslide.OpenSlide(wsi_path)

    # Calculate the downsampling factor
    downsample = wsi.level_downsamples[level]
    
    # Read the metadata
    metadata = pd.read_csv(metadata_path)
    
    # Get the WSI name from the path
    wsi_name = wsi_path.stem
    print(f"WSI name: {wsi_name}")

    full_name = "camelyon16_" + wsi_name
    print(full_name)
    # Filter metadata for the current WSI
    wsi_metadata = metadata[metadata['wsi'] == full_name]
    
    wsi_metadata.drop(wsi_metadata.columns[0], axis=1)    
    
    print(wsi_metadata)
    
    print("shape: ",wsi_metadata.shape)

    # Get the dimensions of the WSI at the specified level
    width, height = wsi.level_dimensions[level]
    
    # Create an empty heatmap
    heatmap = np.zeros((height, width))
    
    # Iterate through the patches for this WSI
    for index, row in wsi_metadata.iterrows():
        if index in predictions_df.index:
            x = int(row['coord_x'] / downsample)
            y = int(row['coord_y'] / downsample)
            patch_size_level = int(patch_size / downsample)
            
            # Get the prediction for this patch
            prediction = predictions_df.loc[index, 'prediction']
            
            # Add the prediction to the heatmap
            heatmap[y:y+patch_size_level, x:x+patch_size_level] = prediction
    
    # Read the WSI image at the specified level
    wsi_image = wsi.read_region((0, 0), level, (width, height)).convert('RGB')
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Display the WSI image
    ax.imshow(wsi_image)
    
    # Overlay the heatmap
    heatmap_overlay = ax.imshow(heatmap, cmap='coolwarm', interpolation='nearest', alpha=0.6)
    
    # Add a colorbar
    plt.colorbar(heatmap_overlay, ax=ax, label='Tumor Probability')
    
    # Set the title
    plt.title(f'Tumor Heatmap for {wsi_name}')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_path}")

