import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as escnn_nn
from torch.nn import functional as F
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """CNN with hooks for multi‐layer domain alignment."""
    def __init__(
        self,
        num_channels: int = 1,
        num_classes: int = 3,
        input_size: tuple = (100, 100),
    ):
        super().__init__()
        # conv block 1
        self.conv1 = nn.Conv2d(num_channels,  8, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)
        # conv block 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        # conv block 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.2)
        # for converting feature‐maps → vectors for alignment
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # figure out flatten size for fc1
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, *input_size)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flat_dim = x.numel()

        # bottleneck
        self.fc1 = nn.Linear(flat_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        # classifier
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # --- conv block 1 ---
        c1 = F.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(c1)
        p1 = self.dropout(p1)
        a1 = self.gap(p1).view(x.size(0), -1)    # ↪︎  shape (B, 8)

        # --- conv block 2 ---
        c2 = F.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool2(c2)
        p2 = self.dropout(p2)
        a2 = self.gap(p2).view(x.size(0), -1)    # ↪︎  shape (B, 16)

        # --- conv block 3 ---
        c3 = F.relu(self.bn3(self.conv3(p2)))
        p3 = self.pool3(c3)
        p3 = self.dropout(p3)
        a3 = self.gap(p3).view(x.size(0), -1)    # ↪︎  shape (B, 32)

        # --- bottleneck & classifier ---
        flat = p3.view(x.size(0), -1)
        z    = self.fc1(flat)
        z    = self.ln1(z)
        logits = self.fc2(z)

        # return both the list of “alignment” features and the final logits
        return {'features': [a1, a2, a3, z], 'logits': logits}


class ENN(nn.Module):
    """ENN model. Can be equivariant to C_N or D_N. D_4 used for most experiments.

    Args:
        num_channels (int, optional): Number of input channels. Defaults to 1.
        num_classes (int, optional): Number of classes. Defaults to 3.
        input_size (tuple, optional): Input size. Defaults to (100, 100).
        N (int, optional): Number of rotations. Defaults to 4.
        dihedral (bool, optional): Whether to use dihedral group. Defaults to True.
    """
    def __init__(
        self,
        num_channels: int = 1,
        num_classes: int = 3,
        input_size: tuple = (100, 100),
        N=4,
        dihedral=True,
    ):
        super(ENN, self).__init__()

        if N == 1:
            self.r2_act = gspaces.trivialOnR2()  # D1 group and C1 group

        else:
            if dihedral:
                self.r2_act = gspaces.flipRot2dOnR2(
                    N=N
                )  # D4 group with 4 rotations and flip
            else:
                self.r2_act = gspaces.rot2dOnR2(
                    N=N
                )  # D4 group with 4 rotations and flip

        self.input_type = escnn_nn.FieldType(
            self.r2_act, num_channels * [self.r2_act.trivial_repr]
        )
        self.conv1 = escnn_nn.R2Conv(
            in_type=self.input_type,
            out_type=escnn_nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]),
            kernel_size=5,
            padding=2,
        )
        self.bn1 = escnn_nn.InnerBatchNorm(self.conv1.out_type)
        self.relu1 = escnn_nn.ReLU(self.conv1.out_type)
        self.pool1 = escnn_nn.PointwiseMaxPool2D(
            self.conv1.out_type, kernel_size=2, stride=2, padding=0
        )
        self.dropout1 = escnn_nn.PointwiseDropout(self.conv1.out_type, p=0.2)

        self.conv2 = escnn_nn.R2Conv(
            in_type=self.conv1.out_type,
            out_type=escnn_nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
            kernel_size=3,
            padding=1,
        )

        self.bn2 = escnn_nn.InnerBatchNorm(self.conv2.out_type)
        self.relu2 = escnn_nn.ReLU(self.conv2.out_type)
        self.pool2 = escnn_nn.PointwiseMaxPool2D(
            self.conv2.out_type, kernel_size=2, stride=2, padding=0
        )
        self.dropout2 = escnn_nn.PointwiseDropout(self.conv2.out_type, p=0.2)

        self.conv3 = escnn_nn.R2Conv(
            in_type=self.conv2.out_type,
            out_type=escnn_nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            kernel_size=3,
            padding=1,
        )
        self.bn3 = escnn_nn.InnerBatchNorm(self.conv3.out_type)
        self.relu3 = escnn_nn.ReLU(self.conv3.out_type)
        self.pool3 = escnn_nn.PointwiseMaxPool2D(
            self.conv3.out_type, kernel_size=2, stride=2, padding=0
        )
        self.dropout3 = escnn_nn.PointwiseDropout(self.conv3.out_type, p=0.2)

        self.gpool = escnn_nn.GroupPooling(self.pool3.out_type)

        c = self.gpool.out_type.size
        dummy_input = torch.zeros(1, num_channels, *input_size)
        dummy_input = escnn_nn.GeometricTensor(dummy_input, self.input_type)
        with torch.no_grad():
            dummy_output = self.gpool(
                self.pool3(
                    self.relu3(
                        self.bn3(
                            self.conv3(
                                self.pool2(
                                    self.relu2(
                                        self.bn2(
                                            self.conv2(
                                                self.pool1(
                                                    self.relu1(
                                                        self.bn1(
                                                            self.conv1(dummy_input)
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        flattened_size = dummy_output.tensor.view(1, -1).shape[1]

        self.fc1 = nn.Linear(in_features=flattened_size, out_features=256)
        self.fc1.weight.data.normal_(0, 0.005)
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


##############################################################################################


def cnn_shapes():
    model = CNN(num_channels=1, num_classes=3, input_size=(100, 100))
    return model


def cnn_astro_objects():
    model = CNN(num_channels=1, num_classes=3, input_size=(100, 100))
    return model


def cnn_mnistm():
    model = CNN(num_channels=3, num_classes=10, input_size=(32, 32))
    return model


def cnn_gzevo():
    model = CNN(num_channels=3, num_classes=6, input_size=(100, 100))
    return model


def d4_shapes():
    model = ENN(
        num_channels=1, num_classes=3, N=4, dihedral=True, input_size=(100, 100)
    )
    return model


def d4_astro_objects():
    model = ENN(
        num_channels=1, num_classes=3, N=4, dihedral=True, input_size=(100, 100)
    )
    return model


def d4_mnistm():
    model = ENN(num_channels=3, num_classes=10, N=4, dihedral=True, input_size=(32, 32))
    return model


def d4_gzevo():
    model = ENN(
        num_channels=3, num_classes=6, N=4, dihedral=True, input_size=(100, 100)
    )
    return model

def cnn_mrssc2():
    model = CNN(num_channels=3, num_classes=7, input_size=(100, 100))
    return model

def d4_mrssc2():
    model = ENN(num_channels=3, num_classes=7, N=4, dihedral=True, input_size = (100, 100))
    return model


## other order D_N models can be constructed by specifcying dihedral = True with varying N
## cyclic group models can be constructed by specifying dihedral = False with varying N

shapes_models = {"cnn": cnn_shapes, "d4": d4_shapes}
astro_objects_models = {"cnn": cnn_astro_objects, "d4": d4_astro_objects}
mnistm_models = {"cnn": cnn_mnistm, "d4": d4_mnistm}
gz_evo_models = {"cnn": cnn_gzevo, "d4": d4_gzevo}
mrssc2_models = {"cnn": cnn_mrssc2, "d4": d4_mrssc2}

model_dict = {
    "shapes": shapes_models,
    "astro_objects": astro_objects_models,
    "mnist_m": mnistm_models,
    "gz_evo": gz_evo_models,
    "mrssc2": mrssc2_models
}

if __name__ == "__main__":
    # model = ENN(num_channels=3, num_classes=10, N=4, dihedral=True, input_size=(28, 28))
    model = CNN(num_channels=3, num_classes=10)
    summary(model, (3, 28, 28))
