'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2B of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			GG_3895
# Author List:		Ashwin Agrawal, Siddhant Godbole, Soham Pawar, Aditya Waghmare
# Filename:			task_2b_model_training.py
###################################################################################################


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define a preprocessing function for training images
def preprocess_train_image(image):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)

# Define a preprocessing function for testing images
def preprocess_test_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Resize both width and height to 256
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)


# Path to the root folder containing class folders for training dataset
train_data_dir = 'datasets/dataset'
# Path to the root folder containing class folders for test dataset
test_data_dir = 'datasets/test_dataset'

# Map the class labels to their corresponding folder names
class_to_label = {
    "combat": 0,
    "humanitarianaid": 1,
    "militaryvehicles": 2,
    "fire": 3,
    "destroyedbuilding": 4
}

# Create a custom dataset using ImageFolder and apply the class_to_label mapping for training dataset
custom_dataset = ImageFolder(root=train_data_dir, transform=preprocess_train_image)

# Replace the class labels in the dataset with numerical labels using the mapping
modified_samples = []
for item in custom_dataset.samples:
    path, label = item
    folder_name = os.path.basename(os.path.dirname(path))
    numerical_label = class_to_label.get(folder_name, -1)
    if numerical_label != -1:
        modified_samples.append((path, numerical_label))

# Update the dataset samples with the modified list for training dataset
custom_dataset.samples = modified_samples

# Create DataLoader for training dataset
train_loader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

# Create a custom dataset for testing using ImageFolder and apply the class_to_label mapping for test dataset
test_dataset = ImageFolder(root=test_data_dir, transform=preprocess_test_image)

# Replace the class labels in the test dataset with numerical labels using the mapping
modified_test_samples = []
for item in test_dataset.samples:
    path, label = item
    folder_name = os.path.basename(os.path.dirname(path))
    numerical_label = class_to_label.get(folder_name, -1)
    if numerical_label != -1:
        modified_test_samples.append((path, numerical_label))

# Update the test dataset samples with the modified list for testing dataset
test_dataset.samples = modified_test_samples

# Create DataLoader for test dataset
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the model architecture
eff_net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)

num_features = eff_net._fc.in_features

eff_net._fc = nn.Linear(num_features, 5)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(eff_net._fc.parameters(), lr=0.001)

num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eff_net.to(device)

# Training loop
for epoch in range(num_epochs):
    eff_net.train()
    total_correct = 0
    total_samples = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = eff_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    train_accuracy = total_correct / total_samples * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.2f}%')

# Evaluation
eff_net.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = eff_net(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
torch.save(eff_net, 'trained_model.pth')
