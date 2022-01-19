import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
# from vit_pytorch.efficient import ViT
from vit_pytorch.vit import ViT
from get_dataset import my_dataset

device = 'cuda'


# Training settings
valid_epoch = 1
batch_size = 64
epochs = 20
lr = 0.0001
gamma = 0.7
seed = 42


model = ViT(image_size=224,
            patch_size=32,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1).to(device)
model.train()

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = my_dataset(root='/data/dongkyu/ILSVRC2012/train', transforms=transforms)
valset = my_dataset(root='/data/dongkyu/ILSVRC2012/val', transforms=transforms)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, label, label_idx in tepoch:
            data = data.to(device)
            label = label.to(device)
            label_idx = label_idx.to(device)

            output = model(data)
            loss = criterion(output, label)
            print(f"output:{output}, shape:{output.shape}")
            print(f"label:{label_idx}, shape:{label_idx.shape}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label_idx).float().mean()
            step_loss = loss
            epoch_loss += step_loss
            tepoch.set_postfix(Loss='{:.6f}'.format(epoch_loss), Acc='{:.4f}'.format(acc))

        if (epoch + 1) % valid_epoch == 0:
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label, label_idx in valid_loader:
                    data = data.to(device)
                    label = label.to(device)
                    label_idx = label_idx.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (output.argmax(dim=1) == label_idx).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )