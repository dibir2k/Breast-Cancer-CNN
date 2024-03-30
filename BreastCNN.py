import torch.nn as nn
import torch


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

#alexnet

class BreastCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # o = floor( (n + 2P - m) / s ) + 1
        # P = 1
        # input n = 128 X 128
        # Kernel Size m = 3
        # Stride s = 1 Default

        self.input = nn.Sequential(
#227
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(p=0.5),
#26
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(p=0.5),
#26
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(p=0.5),
#3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=16),
            #nn.Dropout(p=0.5),

            nn.Flatten()
        )

        self.dense = nn.Sequential(

            #nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        output=self.input(x)

        output=self.dense(output)

        return output






# self.input = nn.Sequential(
# #227
#             nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
#             #nn.BatchNorm2d(96),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             #nn.Dropout(p=0.5),
# #26
#             nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
#             #nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             #nn.Dropout(p=0.5),
# #26
#             nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
#             #nn.BatchNorm2d(384),
#             nn.ReLU(),
#             #nn.MaxPool2d(kernel_size=2),
#             #nn.Dropout(p=0.5),
# #3
#             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
#             #nn.BatchNorm2d(384),
#             nn.ReLU(),
#             #nn.AvgPool2d(kernel_size=2),
#             #nn.Dropout(p=0.5),

#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
#             #nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=3, stride=2),

#             # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=0),
#             # nn.BatchNorm2d(128),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=3, stride=2),

#             nn.Flatten()
#         )

#         self.dense = nn.Sequential(

#             #nn.Dropout(p=0.5),
#             nn.Linear(9216, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, 1000),
#             nn.ReLU(),
#             # nn.ReLU(),
#             nn.Linear(1000, 3)
#         )