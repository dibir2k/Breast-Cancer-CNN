from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
print(preprocess)

class MyResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier_layer = nn.Sequential(
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(1000 , 512),
            #nn.BatchNorm1d(512),
            #nn.Dropout(p=0.2),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(512 , 3)
        )

    def forward(self, x):
        x = self.model._forward_impl(x)
        x = self.classifier_layer(x)

        return x
    
my_model = MyResnet()

print(my_model)