#import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# PS D:\CAMELYON16\camelyon_dissertation> & D:/Userssgnetanaconda3/envs/camelyon_venv/python.exe d:/CAMELYON16/camelyon_dissertation/src/training_models/googlenet_model.py
# Traceback (most recent call last):
#   File "d:\CAMELYON16\camelyon_dissertation\src\training_models\googlenet_model.py", line 5, in <module>
#     from torchvision import models, transforms
#   File "D:\Userssgnetanaconda3\envs\camelyon_venv\Lib\site-packages\torchvision\__init__.py", line 10, in <module>
#     from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\Userssgnetanaconda3\envs\camelyon_venv\Lib\site-packages\torchvision\datasets\__init__.py", line 1, in <module>
#     from ._optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel
#   File "D:\Userssgnetanaconda3\envs\camelyon_venv\Lib\site-packages\torchvision\datasets\_optical_flow.py", line 10, in <module>
#     from PIL import Image
#   File "D:\Userssgnetanaconda3\envs\camelyon_venv\Lib\site-packages\PIL\Image.py", line 100, in <module>
#     from . import _imaging as core
# ImportError: DLL load failed while importing _imaging: The operating system cannot run %1.

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

def get_loss_and_optimizer(model):

    """
    Setup the loss function and optimizer.

    :param model: The model to be trained.
    :return: Loss function and optimizer.
    """

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    return loss, optimizer





