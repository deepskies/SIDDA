import torch
import torch.nn as nn
from torch.nn import functional as F
from escnn import gspaces
from escnn import nn as escnn_nn
import cnn
import torchvision
from torchsummary import summary

num_classes = 10
feature_fields = [12, 24, 48, 48, 48, 48, 96, 96, 96, 112, 192]    

class ConvBlock(nn.Module):
    def __init__(self,
                 in_type: escnn_nn.FieldType, 
                 out_type: escnn_nn.FieldType, 
                 kernel_size: int, 
                 padding: int, 
                 stride: int, 
                 bias: bool, 
                 mask_module: bool = False
            ):
        
        super(ConvBlock, self).__init__()
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
            self.mask = escnn_nn.MaskModule(in_type, 255, margin=1)
    
    def forward(self, x):
        if self.mask_module:
            x = self.mask(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class GeneralSteerableCNN(torch.nn.Module):
    
    def __init__(
        self, N, n_classes=num_classes, 
        feature_fields = feature_fields, reflections = False, maximum_frequency = None
    ):
        super(GeneralSteerableCNN, self).__init__()
        
        self.N = N
        if reflections:
            self.r2_act = gspaces.flip2dOnR2() if self.N == 1 else gspaces.flipRot2dOnR2(N=self.N)
        else:
            self.r2_act = gspaces.rot2dOnR2(N=self.N)

        in_type = escnn_nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        self.input_type = in_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[0]*[self.r2_act.regular_repr])
        
        self.block1 = ConvBlock(in_type, out_type, kernel_size=3, padding=2, stride=2, bias=False, mask_module=True)
        in_type = self.block1.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[1]*[self.r2_act.regular_repr])
        self.block2 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool1 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block2.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[2]*[self.r2_act.regular_repr])
        self.block3 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block3.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[3]*[self.r2_act.regular_repr])
        self.block4 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool2 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block4.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[4]*[self.r2_act.regular_repr])
        self.block5 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block5.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[5]*[self.r2_act.regular_repr])
        self.block6 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block6.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[6]*[self.r2_act.regular_repr])
        self.block7 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool3 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block7.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[7]*[self.r2_act.regular_repr])
        self.block8 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        
        in_type = self.block8.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[8]*[self.r2_act.regular_repr])
        self.block9 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool4 = escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)

        in_type = self.block9.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[9]*[self.r2_act.regular_repr])
        self.block10 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block10.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[10]*[self.r2_act.regular_repr])
        self.block11 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool5 = escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)

        self.gpool = escnn_nn.GroupPooling(out_type)
        
        # number of output channels
        # b, c, h, w = self.gpool.evaluate_output_shape(self.pool3.out_type)
        # d = c*h*w
        c = self.gpool.out_type.size
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(25*c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(32, n_classes),
        )
    
    def forward(self, input: torch.Tensor):
        x = escnn_nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool3(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.pool4(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.pool5(x)
        x = self.gpool(x)
        features = x.tensor.squeeze(-1).squeeze(-1) 
        x = x.tensor
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return features, x


def load_d1():
    D1_model = GeneralSteerableCNN(N=1,reflections=True)
    return D1_model

def load_d2():
    D2_model = GeneralSteerableCNN(N=2,reflections=True)
    return D2_model

def load_d4():
    D4_model = GeneralSteerableCNN(N=4,reflections=True)
    return D4_model

def load_d8():
    D8_model = GeneralSteerableCNN(N=8,reflections=True)
    return D8_model

def load_d16():
    D16_model = GeneralSteerableCNN(N=16,reflections=True)
    return D16_model

def load_d32():
    D32_model = GeneralSteerableCNN(N=32,reflections=True)
    return D32_model

def load_d64():
    D64_model = GeneralSteerableCNN(N=64,reflections=True)
    return D64_model

def load_c1():
    C1_model = GeneralSteerableCNN(N=1)
    return C1_model

def load_c2():
    C2_model = GeneralSteerableCNN(N=2)
    return C2_model

def load_c4():
    C4_model = GeneralSteerableCNN(N=4)
    return C4_model

def load_c8():
    C8_model = GeneralSteerableCNN(N=8)
    return C8_model 

def load_c16():
    C16_model = GeneralSteerableCNN(N=16)
    return C16_model


model_dict = {
    'D1': load_d1,
    'D2': load_d2,
    'D4': load_d4,
    'D8': load_d8,
    'D16': load_d16,
    'D32': load_d32,
    'D64': load_d64,
    'C1': load_c1,
    'C2': load_c2,
    'C4': load_c4,
    'C8': load_c8,
    'C16': load_c16
}

if __name__ == '__main__':
    model = model_dict['D8']()
    model.eval()
    
    summary(model, (3, 255, 255))