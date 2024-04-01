#import pandas as pd
import re
from PIL import Image
import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import optuna

BATCHSIZE = 16
CLASSES = 3
N_TRAIN_EXAMPLES = BATCHSIZE * 50
N_VALID_EXAMPLES = BATCHSIZE * 20


#function to order datasets
def myOrder(file):
    num = int(re.search(r'\d+', file).group())
    return num

def groupFiles(files):
    file = files[0]
    i = 0
    j = 0
    grouped_files = []
    while myOrder(files[i]) < myOrder(files[-1]):
        j = i + 1
        group = []
        group.append(files[i])
        while myOrder(files[j]) == myOrder(files[i]):
            if "mask" not in files[j]:
                group.insert(0,files[j])
            else: group.append(files[j])
            j = j + 1
        grouped_files.append(group)
        i = j
    
    return grouped_files

#Select region of interest from each file
def regionOfInterest(files, labels):
    final_images = []
    for i in range(len(files)):
        if "normal" in files[i][0]: 
            final_images.append(cv.imread(files[i][0], cv.IMREAD_GRAYSCALE))
            continue

        img = cv.imread(files[i][0], cv.IMREAD_GRAYSCALE)
        updated_labels = labels

        for j in range(1,len(files[i])):
            mask = cv.imread(files[i][j], cv.IMREAD_GRAYSCALE)
            cv.imshow("mask", mask)
            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            bounding_boxes = [cv.boundingRect(contour) for contour in contours]
            cropped_imgs = [img[y:y+h, x:x+w] for (x,y,w,h) in bounding_boxes]
            final_images.extend(cropped_imgs)

            if "benign" in files[i][0]:
                updated_labels.insert(i, 0)
            elif "malignant" in files[i][0]:
                updated_labels.insert(i, 1)

    final_images = [Image.fromarray(cv_image) for cv_image in final_images]

    return final_images, updated_labels

#Function for contrast enhancement
def contrastEnhancement(img):
    img = np.array(img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    # Top Hat Transform
    topHat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    # Black Hat Transform
    blackHat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    res = img - topHat + blackHat

    res = Image.fromarray(np.uint8(res)).convert('L')

    return res



#read files to image
def readFiles(files):
    return [Image.open(fname).convert("L") for fname in files]

#read files to image
def readFilesRGB(files):
    return [Image.open(fname).convert("RGB") for fname in files]

#Alternative mean and std
def mean_std(imgs):
    num_samples = len(imgs)
    mean = [0.]*1
    stdTemp = [0.]*1
    for i in range(len(imgs)):
        img = imgs[i]
        img = np.array(img).astype('float64') / 255.
        mean[0] += np.mean(img)

    mean = [(m/num_samples) for m in mean]

    for i in range(num_samples):
        img = imgs[i]
        img = np.array(img).astype('float64') / 255.
        stdTemp[0] += ((img - mean[0])**2).sum()/(img.shape[0]*img.shape[1])

    std = [np.sqrt(s/num_samples) for s in stdTemp]

    return mean, std

#Get mean and std of dataset
def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_batches = 0
    mean = 0.0
    std = 0.0
    num_pixels = 0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        num_batches += 1
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_batches
    std /= num_batches

    return mean, std

def train(model, num_epochs, train_dl, valid_dl, trial):
    loss_fn = nn.CrossEntropyLoss()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for imgs_batch, labels_batch in train_dl:
            pred = model(imgs_batch)
            loss = loss_fn(pred, labels_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*labels_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == labels_batch).float() # dim = 1 along columns
            accuracy_hist_train[epoch] += is_correct.sum()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()

        with torch.no_grad():
            for imgs_batch, labels_batch in valid_dl:
                pred = model(imgs_batch)
                loss = loss_fn(pred, labels_batch)
                #scheduler.step(loss)
                loss_hist_valid[epoch] += loss.item()*labels_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == labels_batch).float() # dim = 1 along columns
                accuracy_hist_valid[epoch] += is_correct.sum()
        
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch + 1} accuracy: '
              f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
              f'{accuracy_hist_valid[epoch]:4f}')
        
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

def objective(model, num_epochs, train_dl, valid_dl, trial):
    loss_fn = nn.CrossEntropyLoss()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    accuracy = 0.
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (imgs_batch, labels_batch) in enumerate(train_dl):
            if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                break
            pred = model(imgs_batch)
            loss = loss_fn(pred, labels_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            is_correct = (torch.argmax(pred, dim=1) == labels_batch).float() 
            accuracy_hist_train[epoch] += is_correct.sum()

        accuracy_hist_train[epoch] /= min(len(train_dl.dataset), N_TRAIN_EXAMPLES)

        model.eval()

        with torch.no_grad():
            for batch_idx, (imgs_batch, labels_batch) in enumerate(valid_dl):
                pred = model(imgs_batch)
                loss = loss_fn(pred, labels_batch)
                loss_hist_valid[epoch] += loss.item()*labels_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == labels_batch).float() 
                accuracy_hist_valid[epoch] += is_correct.sum()
        
        loss_hist_valid[epoch] /= min(len(valid_dl.dataset), N_VALID_EXAMPLES)
        accuracy_hist_valid[epoch] /= min(len(valid_dl.dataset), N_VALID_EXAMPLES)

        accuracy = accuracy_hist_valid[epoch]
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f'Epoch {epoch + 1} accuracy: '
              f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
              f'{accuracy_hist_valid[epoch]:4f}')
        
    return accuracy