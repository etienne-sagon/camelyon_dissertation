from skimage.filters import threshold_otsu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
from pathlib import Path
import cv2
import os.path as osp
import glob
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import openslide
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_patches_from_slide(slide_path, level=1):
    #https://github.com/3dimaging/DeepLearningCamelyon/blob/master/4%20-%20Prediction%20and%20Evaluation/Prediction_googlenet.py
    
    # Your provided function here
    # (I'm including the full function for completeness)
    print(slide_path)
    dimensions = []
    
    with openslide.open_slide(slide_path) as slide:
        level_dimensions = slide.level_dimensions[level]
        dtotal = (level_dimensions[0] / 224, level_dimensions[1] / 224)
        thumbnail = slide.get_thumbnail((dtotal[0], dtotal[1]))
        thum = np.array(thumbnail)
        ddtotal = thum.shape
        dimensions.extend(ddtotal)
        hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
        minhsv = np.array([hthresh, sthresh, 70], np.uint8)
        maxhsv = np.array([180, 255, vthresh], np.uint8)
        thresh = [minhsv, maxhsv]
        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
        contours, _ = cv2.findContours(rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
        bboxt = pd.DataFrame(columns=bboxtcols)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            bboxt = pd.concat([bboxt, pd.DataFrame([[x, x+w, y, y+h]], columns=bboxtcols)], ignore_index=True)
        bboxt = pd.DataFrame(bboxt)
        
        xxmin = bboxt['xmin'].values
        xxmax = bboxt['xmax'].values
        yymin = bboxt['ymin'].values
        yymax = bboxt['ymax'].values
        xxxmin = np.min(xxmin)
        xxxmax = np.max(xxmax)
        yyymin = np.min(yymin)
        yyymax = np.max(yymax)
        dcoord = (xxxmin, xxxmax, yyymin, yyymax)
        dimensions.extend(dcoord)
       
        samplesnew = pd.DataFrame(np.array(thumbnail.convert('L')))
        #print(samplesnew)
        samplesforpred = samplesnew.loc[yyymin:yyymax, xxxmin:xxxmax]
        dsample = samplesforpred.shape
        dimensions.extend(dsample)
        np.save(f'dimensions_{osp.splitext(osp.basename(slide_path))[0]}', dimensions)
        #print(samplesforpred)
        samplesforpredfinal = pd.DataFrame(samplesforpred.stack())
        #print(samplesforpredfinal)
        samplesforpredfinal['tile_loc'] = list(samplesforpredfinal.index)
        samplesforpredfinal.reset_index(inplace=True, drop=True)
        samplesforpredfinal['slide_path'] = slide_path
        #print(samplesforpredfinal)

    return samplesforpredfinal

def get_googlenet_model(num_classes=2, pretrained=False):

    model = models.googlenet(weights=None, aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Load your GoogLeNet model
def load_model(model_path):

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
    print("model loaded")
    return model

# Function to extract and preprocess a patch
def extract_and_preprocess_patch(slide, coords, patch_size=224, level = 1):
    x, y = coords
    patch = slide.read_region((x * patch_size, y * patch_size), level, (patch_size, patch_size)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(patch).unsqueeze(0)

# Function to make prediction for a single patch
def predict_patch(model, patch):
    with torch.no_grad():
        output = model(patch)
        probabilities = torch.softmax(output, dim=1)
    return probabilities[0, 1].item()

# Main function to process a slide and create a heatmap
def create_slide_heatmap(slide_path, model_path, level=1):
    # Load the slide and model
    slide = openslide.OpenSlide(slide_path)
    model = load_model(model_path)
    
    # Get patch locations
    patch_df = find_patches_from_slide(slide_path)
    
    # Create empty heatmap
    level_dimensions = slide.level_dimensions[level]
    heatmap = np.zeros((level_dimensions[1] // 224, level_dimensions[0] // 224))
    max_y, max_x = heatmap.shape
    
    # Process each patch
    for _, row in tqdm(patch_df.iterrows(), total=len(patch_df)):
        
        coords = row['tile_loc']
        if 0 <= coords[1] < max_y and 0 <= coords[0] < max_x:
            patch = extract_and_preprocess_patch(slide, coords, level=level)
            prediction = predict_patch(model, patch)
            # Update heatmap
            heatmap[coords[1], coords[0]] = prediction
    return heatmap

# Visualize and save the heatmap
def visualize_heatmap(heatmap, output_path):
    plt.figure(figsize=(20, 20))
    plt.imshow(heatmap, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Tumor Probability Heatmap')
    plt.savefig(output_path)
    plt.close()

