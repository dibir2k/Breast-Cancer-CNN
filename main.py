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
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import numpy as np

#import helpfunctions
from helpFunctions import myOrder, groupFiles, readFiles, train, mean_std, objective

#import Breast Data Set class BreastCNN and TRAIN
from BreastDataSet import BreastDataSet
from BreastCNN import BreastCNN

DEVICE = torch.device("cpu")
BATCHSIZE = 32
CLASSES = 3
DIR = os.getcwd()
EPOCHS = 100
LR = 0.001


#kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset

#Download dataset from kaggle
if not os.path.exists('./breast-ultrasound-images-dataset'):
    dataset = "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset"

    od.download(dataset)

data_dir = DIR + "/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"

#benign data
benign_dir = data_dir + "/benign"
benign_files = [benign_dir + "/" + f for f in os.listdir(benign_dir) if "mask" not in f]
benign_files.sort(key=myOrder)
#grouped_bfiles = groupFiles(benign_files)
labels = [1]*len(benign_files)


#malignant data
malignant_dir = data_dir + "/malignant"
malignant_files = [malignant_dir + "/" + f for f in os.listdir(malignant_dir) if "mask" not in f]
malignant_files.sort(key=myOrder)
#grouped_mfiles = groupFiles(malignant_files)
labels_m = [2]*len(malignant_files)


#normal data
normal_dir = data_dir + "/normal"
normal_files = [normal_dir + "/" + f for f in os.listdir(normal_dir) if "mask" not in f]
normal_files.sort(key=myOrder)
#grouped_nrmfiles = groupFiles(normal_files)
labels_n = [0]*len(normal_files)

#Create list of all files
breast_files = benign_files
breast_files.extend(malignant_files)
breast_files.extend(normal_files)

#labels
labels.extend(labels_m)
labels.extend(labels_n)

breast_files_train, breast_files_test, labels_train, labels_test = train_test_split(breast_files, labels, stratify=labels, 
                                                                                    test_size=0.15, shuffle=True)

breast_files_train, breast_files_valid, labels_train, labels_valid = train_test_split(breast_files_train, labels_train, stratify=labels_train, 
                                                                                    test_size=0.2, shuffle=True)


breast_images_train = readFiles(breast_files_train)
breast_images_valid = readFiles(breast_files_valid)
breast_images_test = readFiles(breast_files_test)

mean2, std2 = mean_std(breast_images_train)

print(" mean 2: ", mean2, "std 2: ", std2)

#Data Augmentation for Training set
train_transform = v2.Compose([
    v2.RandomResizedCrop(227),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(60),
    v2.RandomVerticalFlip(),
    v2.Resize([256,256]),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize(mean=mean2, std=std2)
])

#For Validation
valid_transform = v2.Compose([
    v2.Resize([256,256]),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize(mean=mean2, std=std2)
])

train_dataset = BreastDataSet(breast_images_train, labels_train, transform=train_transform)
valid_dataset = BreastDataSet(breast_images_valid, labels_valid, transform=valid_transform)
test_dataset = BreastDataSet(breast_images_test, labels_test, transform=valid_transform)


if __name__ == "__main__":
    batch_size = BATCHSIZE
    num_epochs = EPOCHS
    learning_rate = LR #small batch sizes require small learning rates

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_dataset, batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(BreastCNN(trial), num_epochs, train_dl, valid_dl, trial), n_trials=200, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

   
    model = BreastCNN(trial)

    hist = train(model, num_epochs, train_dl, valid_dl, trial)

    torch.save(model.state_dict(),'./saved_model/breastCancerDetection.pth')

    model = BreastCNN(trial)
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

    x_arr = np.arange(len(hist[0])) + 1
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1,2,1)
    ax.plot(x_arr, hist[0], "-o", label = "Train acc.")
    ax.plot(x_arr, hist[1], "--<", label = "Validation acc.")
    ax.legend(fontsize=15)
    ax.set_xlabel("Epoch", size=15)
    ax.set_ylabel("Loss", size=15)
    ax = fig.add_subplot(1,2,2)
    ax.plot(x_arr, hist[2], "-o", label = "Train loss")
    ax.plot(x_arr, hist[3], "-o", label = "Train loss")
    ax.legend(fontsize=15)
    ax.set_xlabel("Epoch", size=15)
    ax.set_ylabel("Accuracy", size=15)
    fig.savefig("/Figures/Loss-Accuracy.png")