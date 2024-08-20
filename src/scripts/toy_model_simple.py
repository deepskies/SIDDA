import torch
import torch.nn as nn
from torch.nn import functional as F
from escnn import gspaces
from escnn import nn as escnn_nn
import torchvision
from torchsummary import summary
from torchvision import transforms
import numpy as np

num_classes = 3
feature_fields = [2, 2, 2]  

class SimplifiedConvBlock(nn.Module):
    def __init__(self,
                 in_type: escnn_nn.FieldType, 
                 out_type: escnn_nn.FieldType, 
                 kernel_size: int, 
                 padding: int, 
                 stride: int, 
                 bias: bool, 
                 mask_module: bool = False
            ):
        
        super(SimplifiedConvBlock, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.conv = escnn_nn.R2Conv(
            in_type, out_type, kernel_size=kernel_size, 
            padding=padding, stride=stride, bias=bias
        )
        self.bn = escnn_nn.InnerBatchNorm(out_type)
        self.act = escnn_nn.ReLU(out_type, inplace=True)
        self.mask_module = mask_module
        if mask_module:
            self.mask = escnn_nn.MaskModule(in_type, 100, margin=1)
    
    def forward(self, x):
        if self.mask_module:
            x = self.mask(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SimplifiedSteerableCNN(torch.nn.Module):
    
    def __init__(
        self, N, num_classes=num_classes, 
        feature_fields = feature_fields, reflections = False, maximum_frequency = None
    ):
        super(SimplifiedSteerableCNN, self).__init__()
        
        self.N = N
        if reflections:
            self.r2_act = gspaces.flip2dOnR2() if self.N == 1 else gspaces.flipRot2dOnR2(N=self.N)
        else:
            self.r2_act = gspaces.rot2dOnR2(N=self.N)

        in_type = escnn_nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        self.input_type = in_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[0]*[self.r2_act.regular_repr])
        
        self.block1 = SimplifiedConvBlock(in_type, out_type, kernel_size=3, padding=2, stride=2, bias=False, mask_module=True)
        in_type = self.block1.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[1]*[self.r2_act.regular_repr])
        self.block2 = SimplifiedConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        
        self.pool1 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block2.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[2]*[self.r2_act.regular_repr])
        self.block3 = SimplifiedConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        
        self.pool2 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        self.gpool = escnn_nn.GroupPooling(out_type)
        
        c = self.gpool.out_type.size
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(169*c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(32, num_classes),
        )
    
    def forward(self, input: torch.Tensor):
        x = escnn_nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.pool2(x)
        x = self.gpool(x)
        x = x.tensor.view(x.tensor.size(0), -1)
        features = x
        x = self.fully_net(x.reshape(x.shape[0], -1))
        return features, x
    
    
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 100 * 100, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Adjusting the view operation for the new image size
        x = x.view(-1, 32 * 100 * 100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
def cnn():
    model = CNN(num_classes=num_classes)
    return model


def d4_model():
    model = SimplifiedSteerableCNN(N=4,reflections=True, num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = d4_model()
    print(summary(model, (1, 100, 100)))
    x = torch.randn(32, 1, 100 ,100)
    output = model(x)
    print(output[0].shape)
    print(output[1].shape)
