#import necessary libraries
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import src.utils as utils
import src.training_models.googlenet_model as model
import time

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
train_data = datasets.ImageFolder(utils.train_patch_dir, model.train_transform)
val_data = datasets.ImageFolder(utils.val_patch_dir, model.val_transform)


batch_size = utils.batch_size
num_epochs = utils.nb_epochs

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(val_data, 
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

googlenet_model = model.googlenet_model
googlenet_model = googlenet_model.to(device)

# Initialize lists to store the training and validation losses
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    googlenet_model.train()
    epoch_start_time = time.time()  # Start time of the epoch
    
    total_batch = len(train_data) // batch_size
    train_loss = 0.0

    for i, (batch_images, batch_labels) in enumerate(train_loader):
        
        X = batch_images.to(device)
        Y = batch_labels.to(device)

        pre = googlenet_model(X)
        cost = model.loss(pre, Y)

        model.optimizer.zero_grad()
        cost.backward()
        model.optimizer.step()

        train_loss += cost.item()

        if (i+1) % 5 == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, total_batch, cost.item()))

    train_losses.append(train_loss / len(train_loader))

    # Validation phase
    googlenet_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_outputs = googlenet_model(val_images)
            val_cost = model.loss(val_outputs, val_labels)
            
            val_loss += val_cost.item()
    
    val_losses.append(val_loss / len(val_loader))



    epoch_end_time = time.time()  # End time of the epoch
    epoch_duration = epoch_end_time - epoch_start_time  # Duration of the epoch
    print('Epoch [%d/%d] completed in %.2f seconds' % (epoch + 1, num_epochs, epoch_duration))
    print('Training Loss: %.4f, Validation Loss: %.4f' % (train_losses[-1], val_losses[-1]))

# Save model
torch.save(googlenet_model.state_dict(), "D:/CAMELYON16/camelyon_dissertation/models/googlenet_test")