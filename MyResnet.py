from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
print(preprocess)

CLASSES = 3


class MyResnet(nn.Module):
    def __init__(self, trial):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False

        n_layers = trial.suggest_int("n_layers", 1, 4)
        layers = []
        in_features = 1000
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 32, 512)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.4)
            layers.append(nn.Dropout(p))
        
            in_features = out_features

        layers.append(nn.Linear(in_features, CLASSES))

        self.classifier_layer = nn.Sequential(*layers)

        # self.classifier_layer = nn.Sequential(
        #     nn.ReLU(),
        #     #nn.Dropout(p=0.5),
        #     nn.Linear(1000 , 512),
        #     #nn.BatchNorm1d(512),
        #     #nn.Dropout(p=0.2),
        #     nn.ReLU(),
        #     #nn.Dropout(p=0.2),
        #     nn.Linear(512 , 3)
        #)

    def forward(self, x):
        x = self.model._forward_impl(x)
        x = self.classifier_layer(x)

        return x