'''
CODE PROVIDED BY

https://github.com/locuslab/TCN

LICENSED UNDER MIT
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pennylane as qml

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_group_norm: bool = True, gn_groups: int = 8):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.gn1 = nn.GroupNorm(num_groups=min(gn_groups, n_outputs), num_channels=n_outputs) if use_group_norm else nn.Identity()
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.gn2 = nn.GroupNorm(num_groups=min(gn_groups, n_outputs), num_channels=n_outputs) if use_group_norm else nn.Identity()
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.gn1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.gn2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.2, use_group_norm: bool = True, gn_groups: int = 8):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.output_channels =  num_channels[-1]
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
                gn_groups=gn_groups,
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Noise2Noise1DTCN(nn.Module):
    def __init__(self, in_channels=1, num_channels=None, kernel_size=3, dropout=0.1, use_group_norm: bool = True, gn_groups: int = 8,
                 n_qubits: int = 6, vqc_depth: int = 2, shots=None):
        super(Noise2Noise1DTCN, self).__init__()
        if num_channels is None:
            # num_channels = [32, 32, 64, 64, 128, 128, 256, 256]
            # num_channels=[8, 8, 16, 16, 32, 32, 64, 64]
            num_channels=[8, 8, 8, 16, 16, 16]
        # Backbone: quantum temporal convolution block only for the last layer
        layers = []
        for i, c_out in enumerate(num_channels):
            c_in = in_channels if i == 0 else num_channels[i-1]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            if i < len(num_channels) - 1:
                layers.append(TemporalBlock(
                    c_in, c_out, kernel_size,
                    stride=1, dilation=dilation_size, padding=padding,
                    dropout=dropout, use_group_norm=use_group_norm, gn_groups=gn_groups
                ))
            else:
                layers.append(QuantumTemporalBlock(
                    c_in, c_out,
                    n_qubits=n_qubits, depth=vqc_depth, shots=shots
                ))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class VQCCell(nn.Module):
    """
    (N, C_in) -> (N, C_out) with a small variational circuit.
    """
    def __init__(self, in_features: int, out_features: int, n_qubits: int = 6, depth: int = 2, shots=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reduce = nn.Linear(in_features, n_qubits)
        # Prefer GPU backend if available; fallback to CPU
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits, shots=shots)
        except Exception:
            dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)
        
        @qml.qnode(dev, interface="torch", diff_method="best")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (depth, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.proj = nn.Linear(n_qubits, out_features)

    def forward(self, f_bt: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.reduce(f_bt))
        q = self.q_layer(z)
        y = self.proj(q)
        return y


class QuantumTemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_qubits: int = 6, depth: int = 2, shots=None):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.vqc = VQCCell(in_features=n_inputs, out_features=n_outputs, n_qubits=n_qubits, depth=depth, shots=shots)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        f_bt = x.permute(0, 2, 1).reshape(B * T, C)
        y_bt = self.vqc(f_bt)
        y = y_bt.view(B, T, self.n_outputs).permute(0, 2, 1)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)
        