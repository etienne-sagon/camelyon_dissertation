#import necessary libraries
import os
import sys
import time
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import shutil
from pathlib import Path
import re


# Adjust the path to include the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import src.training_models.googlenet_model as model
import src.utils as utils

def move_patches_to_validation(source_dir, dest_dir, percentage=0.1):
    """
    Moves a specified percentage of patches from source directory to destination directory.
    
    :param source_dir: Path to the source directory
    :param dest_dir: Path to the destination directory
    :param percentage: Percentage of patches to move (default is 0.1 for 10%)
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all patch files
    patch_files = list(source_dir.glob('*'))

    # Group patches by their base name (slide class, slide ID, and coordinates)
    patch_groups = {}
    for file in patch_files:
        # Extract the base name: everything up to the last underscore
        base_name = '_'.join(file.stem.split('_')[:-1])
        patch_groups.setdefault(base_name, []).append(file)

    
    # Calculate number of files to move
    num_groups_to_move  = int(len(patch_groups) * percentage)
    
    # Randomly select patch groups to move
    groups_to_move = random.sample(list(patch_groups.keys()), num_groups_to_move)
    
    # Move selected patch groups
    total_moved = 0
    for group in groups_to_move:
        for file in patch_groups[group]:
            shutil.move(str(file), str(dest_dir / file.name))
            total_moved += 1
    
    print(f"Moved {total_moved} patches ({num_groups_to_move} groups) from {source_dir} to {dest_dir}")


def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation losses over epochs.

    :param train_losses: List of training losses per epoch.
    :param val_losses: List of validation losses per epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    #plt.xticks(epochs)  # Ensure that all epochs are shown on the x-axis
    plt.savefig('/app/scripts/models/googlenet_loss_plot.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot
    print("Loss plot saved")

def train_googlenet_model():

    """
    Train a GoogLeNet model using image patches, with the ability to monitor 
    training and validation losses and save the trained model.

    :return: None. The function trains a GoogLeNet model and saves the trained model to a file.
    """
    
    train_start_time = time.time()
    # Move patches before loading data
    # move_patches_to_validation(utils.train_normal_patch_path, utils.val_normal_patch_path, 0.10)
    # move_patches_to_validation(utils.train_tumor_patch_path, utils.val_tumor_patch_path, 0.10)


    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_transform, val_transform = model.get_transform()
    train_data = datasets.ImageFolder(utils.train_patch_dir, transform=train_transform)
    val_data = datasets.ImageFolder(utils.val_patch_dir, transform=val_transform)

    batch_size = utils.batch_size
    num_epochs = utils.nb_epochs

    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

    val_loader = DataLoader(val_data, 
                            batch_size=batch_size,
                            shuffle=True, #set to true 
                            drop_last=False)

    # Load model
    googlenet_model = model.get_googlenet_model(num_classes=2, pretrained=True)
    googlenet_model = googlenet_model.to(device)

    # Initialize loss function and optimizer and scheduler
    loss_fn, optimizer, scheduler = model.get_loss_optimizer_scheduler(googlenet_model)


    # Initialize lists to store the training and validation losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        googlenet_model.train()
        epoch_start_time = time.time()  # Start time of the epoch
        
        total_batch = len(train_loader) 
        train_loss = 0.0

        for i, (batch_images, batch_labels) in enumerate(train_loader):
            
            X = batch_images.to(device)
            Y = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = googlenet_model(X)
            loss = loss_fn(outputs, Y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % batch_size == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Iter [{i + 1}/{total_batch}] Loss: {loss.item():.4f}')

        train_losses.append(train_loss / len(train_loader))

        # Validation phase
        googlenet_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = googlenet_model(val_images)
                val_cost = loss_fn(val_outputs, val_labels)
                
                val_loss += val_cost.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        epoch_end_time = time.time()  # End time of the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Duration of the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f} seconds')
        print(f'Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Save the model
    model_path = "/app/scripts/models/googlenet_test.pth"
    torch.save(googlenet_model.state_dict(), model_path)
    train_end_time = time.time()

    training_time = train_end_time - train_start_time #duration of training
    print(f'Model saved to {model_path}')
    print(f"\nTotal training time: {training_time}")
    
    plot_losses(train_losses, val_losses)