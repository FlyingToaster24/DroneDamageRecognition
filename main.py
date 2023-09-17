from training_tools import load_folders
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import Dataset, DataLoader
from training_tools import resize, augment, train_model
import cv2
import numpy as np

FOLDER_NAME = "data/train/"


safe_building = load_folders(FOLDER_NAME, ["0", "1"])
destroyed_building = load_folders(FOLDER_NAME, ["2", "3"])
SIZE = 64


safe_building = augment(safe_building)
destroyed_building = augment(destroyed_building)

random.shuffle(safe_building)
random.shuffle(destroyed_building)

# n = min(len(safe_building), len(destroyed_building))
#
# safe_building = safe_building[:n]
# destroyed_building = destroyed_building[:n]

safe_building = resize(safe_building, SIZE)
destroyed_building = resize(destroyed_building, SIZE)

labels_safe = np.ones((len(safe_building), 1), np.int32)
labels_destroyed = np.zeros((len(destroyed_building), 1), np.int32)

X = safe_building + destroyed_building
y = np.concatenate((labels_safe, labels_destroyed), axis=0)

train_model(X, y, "cnn_piece", batch_size=16, num_epochs=50, dropout_ratio=0.2, patience = 30, learning_rate=1e-4)


