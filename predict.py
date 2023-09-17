import torch
import cv2
from training_tools import resize
import numpy as np

model = torch.load('model.pt')

image_path = "safe.png"
image = [cv2.imread(image_path)]

image = resize(image, 64)
image = np.array(image, dtype=np.uint8)
image = image / 255.0  # scale the input data to [0, 1]
image = np.transpose(image, (0, 3, 1, 2))  # transpose to match PyTorch format

# Convert X to a PyTorch tensor
image = torch.from_numpy(image).float()

# Perform inference
with torch.no_grad():
    output = model(image)

# Get the predicted class
predicted_class_index = torch.argmax(output, dim=1).item()

# Interpret the prediction
class_labels = ['destroyed', 'safe']
predicted_class_label = class_labels[predicted_class_index]

print(f"Predicted Class: {predicted_class_label}")
