import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
import os
from PIL import Image

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

def load_and_preprocess_validation_data(val_normal_patch_path, val_tumor_patch_path, batch_size=32):
    """
    Load and preprocess validation data from the specified directories.

    :param val_normal_patch_path: Path to the directory containing normal validation patches.
    :param val_tumor_patch_path: Path to the directory containing tumor validation patches.
    :param batch_size: The number of samples per batch in the DataLoader. Defaults to 32.
    
    :return: A PyTorch DataLoader object containing the preprocessed image tensors and corresponding labels.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = []
    
    # Load normal patches
    for img_path in os.listdir(val_normal_patch_path):
        img = Image.open(os.path.join(val_normal_patch_path, img_path))
        img_tensor = transform(img)
        dataset.append((img_tensor, 0))  # 0 for normal
    
    # Load tumor patches
    for img_path in os.listdir(val_tumor_patch_path):
        img = Image.open(os.path.join(val_tumor_patch_path, img_path))
        img_tensor = transform(img)
        dataset.append((img_tensor, 1))  # 1 for tumor
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        recalls.append(recall_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    # Find the threshold that maximizes recall
    optimal_threshold_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_index]

    # Plot precision, accuracy, and F1 score for different thresholds
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold - Validation Set')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'metrics_vs_threshold.png')
    plt.close()

    

    #optimal_threshold = 0.94
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
    plt.title('ROC Curve - Validation Set')
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
    plt.title('Precision-Recall curve: AP={0:0.2f} - Validation Set'.format(average_precision))
    plt.savefig(output_dir / 'precision_recall_curve.png')
    plt.close()
    
    print(f"Metrics and curves saved in {output_dir}")
