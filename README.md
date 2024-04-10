# Breast Ultrasound Cancer Detection with CNN

This project implements two classification models, using a Convolutional Neural Network, to determine whether a breast ultrasound image corresponds to a normal/benign/malignant case. 

The dataset used to train the models was downloaded from kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset (you can get detailed information about the dataset in this link). 

The two models consist of:

* A CNN model designed from scratch, BreastCNN. The model's architecture takes inspiration from AlexNet's architecture. The dense layers, however, are optimized using optuna for best parameter search: number of dense layers and nodes of each layer.

* A previously trained resnet50 model, where only the dense layers were adapted to the project's specificity. The dense layers were optimized using optuna as well. With this transfer learning approach, the training part only takes care of the dense layers.

## Requirements

Make sure you have at leat Python 3.8 installed. This project requires the following packages: torch, torchvision, opendatasets, opencv-python, pillow, scikit-learn, matplotlib, optuna and numpy

If you have pip on your pc, you can just download the required package (e.g. optuna) by doing:

```bash
pip install optuna
```

## Structure

The project is organized as follows:

* **BreastCNN.py**: Implements the CNN model simlar to AlexNet
* **MyResnet.py**: Implements the resnet model, where we load the resnet50 model from torchvision.datasets and switch-off gradient computations. We then add a dense set of layers that depends on the best parameters found by optuna.
* **utils.py**: Contains all the helper functions essential for several stages (not all of them are currently being used). The train function is responsible for training the model and the objective function is used when searching for the best parameters with optuna.
* **main.py**: Performs training of the BreastCNN model. It includes all the stages from loading the data, preparing it for the CNN, performing augmentation and training it. Using optuna, a number of trials is run to search for the best parameters. After the best parameters are found, the model is trained with them and its parameters are saved in the folder 'saved_model'. To accelerate optuna's stage, in *utils.py*, you can set N_TRAIN_EXAMPLES and N_VALID_EXAMPLES to be a smaller portion of the batch size
* **main2.py**: Similar to main.py, but takes care of the resnet model.
* **resnet.ipynb**: a notebook version of *main2.py*. Make sure to upload the files to your google drive and mount it before using it on colab, if you wish to use their gpu's or tpu's capabilities.

After training, the plot of Loss and Accuracy vs. EPOCH # is saved in the folder 'Figures'

## Remarks

The current models still require some work. The best yet accuracy obtained for the test dataset was of 70% using the resnet model. However, due to computing power constraints, I have not yet been able to take the most out of optuna's capabilities, since it takes too long per study. 

The next goal is to try to see if the current accuracy can be further improved by only letting optuna perform more trials (like 100 or 200) or if the models require some more fine-tuning.