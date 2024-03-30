#import pandas as pd
import os
import opendatasets as od
from sklearn.model_selection import train_test_split
import re
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import adjust_contrast
import cv2 as cv

#import helpfunctions
from helpFunctions import myOrder, groupFiles, readFilesRGB, get_mean_std, train

#import Breast Data Set class BreastCNN and TRAIN
from BreastDataSet import BreastDataSet
from MyResnet import MyResnet, preprocess


#kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset

#Download dataset from kaggle
if not os.path.exists('./breast-ultrasound-images-dataset'):
    dataset = "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset"

    od.download(dataset)

data_dir = "./breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"

#benign data
benign_dir = data_dir + "/benign"
benign_files = [benign_dir + "/" + f for f in os.listdir(benign_dir)]
benign_files.sort(key=myOrder)
grouped_bfiles = groupFiles(benign_files)
labels = [1]*len(grouped_bfiles)


#malignant data
malignant_dir = data_dir + "/malignant"
malignant_files = [malignant_dir + "/" + f for f in os.listdir(malignant_dir)]
malignant_files.sort(key=myOrder)
grouped_mfiles = groupFiles(malignant_files)
labels_m = [2]*len(grouped_mfiles)


#normal data
normal_dir = data_dir + "/normal"
normal_files = [normal_dir + "/" + f for f in os.listdir(normal_dir)]
normal_files.sort(key=myOrder)
grouped_nrmfiles = groupFiles(normal_files)
labels_n = [0]*len(grouped_nrmfiles)

#Create list of all files
breast_files = grouped_bfiles
breast_files.extend(grouped_mfiles)
breast_files.extend(grouped_nrmfiles)

#labels
labels.extend(labels_m)
labels.extend(labels_n)

breast_files_train, breast_files_test, labels_train, labels_test = train_test_split(breast_files, labels, stratify=labels, 
                                                                                    test_size=0.15, shuffle=True)

breast_files_train, breast_files_valid, labels_train, labels_valid = train_test_split(breast_files_train, labels_train, stratify=labels_train, 
                                                                                    test_size=0.3, shuffle=True)


breast_images_train = readFilesRGB(breast_files_train)
breast_images_valid = readFilesRGB(breast_files_valid)
breast_images_test = readFilesRGB(breast_files_test)

# #To compute mean and std of files
# valid_transform = v2.Compose([
#     v2.Resize([227,227]),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
# ])

# #Create DataSet Objects
# train_ds = BreastDataSet(breast_images_train, labels_train, transform=valid_transform)
# valid_ds = BreastDataSet(breast_images_valid, labels_train, transform=valid_transform)

batch_size = 16

# train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=False)
# valid_dl = DataLoader(valid_ds, batch_size, shuffle=True, drop_last=False)

# mean_train, std_train = get_mean_std(train_dl)
# mean_valid, std_valid = get_mean_std(valid_dl)

# #Data Augmentation for Training set
train_transform = v2.Compose([
    v2.RandomResizedCrop(256),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(180),
    v2.RandomVerticalFlip(),
    # v2.Resize([227,227]),
    # v2.ToImage(),
    # v2.ToDtype(torch.float32, scale=True),
    # v2.Normalize(mean=mean_train, std=std_train)
])

# #For Validation
# valid_transform = v2.Compose([
#     v2.Resize([227,227]),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=mean_valid, std=std_valid)
# ])


train_dataset = BreastDataSet(breast_images_train, labels_train, transform=train_transform)
valid_dataset = BreastDataSet(breast_images_valid, labels_valid, transform=preprocess)
test_dataset = BreastDataSet(breast_images_test, labels_test, transform=preprocess)

if __name__ == "__main__":
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001 #small batch sizes require small learning rates

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_dataset, batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    #CHANGE MODEL TO RESNET
    model = MyResnet()

    train(model, num_epochs, train_dl, valid_dl, learning_rate)

    torch.save(model.state_dict(),'./saved_model/breastCancerDetection.pth')

    model = MyResnet()
    model.load_state_dict(torch.load("./saved_model/breastCancerDetection.pth"))
    accuracy_test = 0
    model.eval()
    with torch.no_grad():
        for imgs_batch, labels_batch in test_dl:
            pred = model(imgs_batch)
            is_correct = (torch.argmax(pred, dim=1) == labels_batch).float()
            accuracy_test += is_correct.sum()
    accuracy_test /= len(test_dl.dataset)
    print(f'Test accuracy: {accuracy_test:.4f}')