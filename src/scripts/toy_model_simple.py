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
    
    
class ConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNet, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2 = nn.Dropout(p=0.2)
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout3 = nn.Dropout(p=0.2)
        
        # Bottleneck Layer (Fully Connected)
        self.fc1 = nn.Linear(in_features=32 * 12 * 12, out_features=256)
        self.fc1.weight.data.normal_(0, .005)
        self.fc1.bias.data.fill_(0.0)
        self.layer_norm = nn.LayerNorm(256)
        
        # Output Layer (Fully Connected)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        # First Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second Convolutional Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third Convolutional Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(-1, 32 * 12 * 12)

        x = self.fc1(x)
        x = self.layer_norm(x)
        latent_space = x

        x = self.fc2(x)
        
        return latent_space, x
    
class D4ConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super(D4ConvNet, self).__init__()

        # D4 Group for 2D images
        self.r2_act = gspaces.flipRot2dOnR2(N=4)  # D4 group with 4 rotations and flip

        # First Convolutional Layer
        self.input_type = escnn_nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.conv1 = escnn_nn.R2Conv(
            in_type=self.input_type,
            out_type=escnn_nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]),
            kernel_size=5,
            padding=2
        )
        self.bn1 = escnn_nn.InnerBatchNorm(self.conv1.out_type)
        self.relu1 = escnn_nn.ReLU(self.conv1.out_type)
        self.pool1 = escnn_nn.PointwiseMaxPool2D(self.conv1.out_type, kernel_size=2, stride=2, padding=0)
        self.dropout1 = escnn_nn.PointwiseDropout(self.conv1.out_type, p=0.2)

        # Second Convolutional Layer
        self.conv2 = escnn_nn.R2Conv(
            in_type=self.conv1.out_type,
            out_type=escnn_nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
            kernel_size=3,
            padding=1
        )

        self.bn2 = escnn_nn.InnerBatchNorm(self.conv2.out_type)
        self.relu2 = escnn_nn.ReLU(self.conv2.out_type)
        self.pool2 = escnn_nn.PointwiseMaxPool2D(self.conv2.out_type, kernel_size=2, stride=2, padding=0)
        self.dropout2 = escnn_nn.PointwiseDropout(self.conv2.out_type, p=0.2)

        # Third Convolutional Layer
        self.conv3 = escnn_nn.R2Conv(
            in_type=self.conv2.out_type,
            out_type=escnn_nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            kernel_size=3,
            padding=1
        )
        self.bn3 = escnn_nn.InnerBatchNorm(self.conv3.out_type)
        self.relu3 = escnn_nn.ReLU(self.conv3.out_type)
        self.pool3 = escnn_nn.PointwiseMaxPool2D(self.conv3.out_type, kernel_size=2, stride=2, padding=0)
        self.dropout3 = escnn_nn.PointwiseDropout(self.conv3.out_type, p=0.2)
        
        self.gpool = escnn_nn.GroupPooling(self.pool3.out_type)
        
        c = self.gpool.out_type.size

        self.fc1 = nn.Linear(in_features=144*c, out_features=256)
        self.fc1.weight.data.normal_(0, .005)
        self.fc1.bias.data.fill_(0.0)
        self.layer_norm = nn.LayerNorm(256)

        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        x = escnn_nn.GeometricTensor(x, self.input_type)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = self.gpool(x)
        
        x = x.tensor.view(x.tensor.size(0), -1)
        x = self.fc1(x)
        x = self.layer_norm(x)
        latent_space = x

        x = self.fc2(x)

        return latent_space, x

def cnn():
    model = ConvNet(num_classes=num_classes)
    return model

def d4_model():
    model = D4ConvNet(num_classes=num_classes)
    return model


if __name__ == "__main__":
    from prettytable import PrettyTable
    
    def print_model_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        
    model = d4_model()
    print_model_parameters(model)
    # print(summary(model, (1, 100, 100)))
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad and p not in model.fc1.parameters() and p not in model.fc2.parameters()))
    x = torch.randn(32, 1, 100 ,100)
    output = model(x)
    print(output[0].shape)
    print(output[1].shape)
