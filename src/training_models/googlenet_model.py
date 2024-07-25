#import necessary libraries
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_googlenet_model(num_classes=2, pretrained=True):

    """
    Load and prepare the GoogLeNet model.

    :param num_classes: Number of classes for the classification task.
    :param pretrained: Whether to load a pre-trained model.
    :return: Modified GoogLeNet model.
    """
    model = models.googlenet(weights="GoogLeNet_Weights.IMAGENET1K_V1" if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Change the final layer for binary classification
    
    return model

def get_transform():

    """
    Returns data transforms for training and validation.

    :return: Dictionary with training and validation transforms.
    """

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def get_loss_optimizer_scheduler(model):

    """
    Setup the loss function and optimizer.

    :param model: The model to be trained.
    :return: Loss function and optimizer.
    """

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, 
                          weight_decay=0.0002)  # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    return loss, optimizer, scheduler





