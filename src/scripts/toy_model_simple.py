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
    
class ConvNet(nn.Module):
    #### for shapes and blobs dataset
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
    
    
class ConvNet_MNISTM(nn.Module):
    #### for shapes and blobs dataset
    def __init__(self, num_classes=3):
        super(ConvNet_MNISTM, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
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
        self.fc1 = nn.Linear(in_features=32 * 8 * 2, out_features=256)
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

        x = x.view(-1, 32 * 8 * 2)

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
    
class ENN_MNISTM(nn.Module):
    def __init__(self, num_classes=3, N=1, dihedral=True):
        super(ENN_MNISTM, self).__init__()
        
        if N==1:
            self.r2_act = gspaces.trivialOnR2()  # D1 group and C1 group
            
        else:
            if dihedral:
                self.r2_act = gspaces.flipRot2dOnR2(N=N)  # D4 group with 4 rotations and flip
            else:
                self.r2_act = gspaces.rot2dOnR2(N=N)  # D4 group with 4 rotations and flip

        # First Convolutional Layer
        self.input_type = escnn_nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
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

        self.fc1 = nn.Linear(in_features=16*c, out_features=256)
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


def cnn(num_classes):
    model = ConvNet(num_classes=num_classes)
    return model

def cnn_mnistm(num_classes):
    model = ConvNet_MNISTM(num_classes=num_classes)
    return model

def d4_model(num_classes):
    model = D4ConvNet(num_classes=num_classes)
    return model

def d1_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=1)
    return model

def d2_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=2)
    return model

def d4_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=4)
    return model

def d8_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=8)
    return model

def c1_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=1, dihedral=False)
    return model

def c2_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=2, dihedral=False)
    return model

def c4_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=4, dihedral=False)
    return model

def c8_mnistm(num_classes):
    model = ENN_MNISTM(num_classes=num_classes, N=8, dihedral=False)
    return model


mnistm_models =  {'c1': c1_mnistm, 'c2': c2_mnistm, 'c4': c4_mnistm, 'c8': c8_mnistm, 'd1': d1_mnistm, 'd2': d2_mnistm, 'd4': d4_mnistm, 'd8': d8_mnistm, 'cnn': cnn_mnistm}
evo_models = {'cnn': cnn, 'd4': d4_model}

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
        
    model = d2_mnistm(num_classes=10)
    print_model_parameters(model)
    # x = torch.randn(32, 1, 100 ,100)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(output[0].shape)
    print(output[1].shape)
