import torch.nn as nn
from torch.nn.utils import weight_norm


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        """
        Temporal Convolutional Network
        Args:
            input_size: Number of input features
            output_size: Number of output features
            num_channels: List of number of channels in the network
            kernel_size: Size of the kernel
            dropout: Dropout rate
        """
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs: Input data dimension (N, C_in, L_in)
        """
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        # print(y1.shape) # torch.Size([B, hidden_dim, L])
        # exit()
        o = self.linear(y1[:, :, -1])
        # return F.log_softmax(o, dim=1)
        return o, y1[:, :, -1]

    def get_activation_maps(self):
        """
        Get the activation maps
        """
        return self.tcn.activation_maps

    def clear_activation_maps(self):
        """
        Clear the activation maps
        """
        self.tcn.activation_maps.clear()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
        Chomp1d
        
        Args:
            chomp_size: Chomp size
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input data

        Returns:
            Cropped data
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        """
        Temporal Block

        Args:
            n_inputs: Number of input features
            n_outputs: Number of output features
            kernel_size: Size of the kernel
            stride: Stride
            dilation: Dilation
            padding: Padding
            dropout: Dropout rate
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input data
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Main Temporal Convolutional Network
        Args:
            num_inputs: Number of input features
            num_channels: List of number of channels in the network
            kernel_size: Size of the kernel
            dropout: Dropout rate
        """
        super(TemporalConvNet, self).__init__()
        self.activation_maps = []  # Instance variable for activations
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    # padding=0,
                    dropout=dropout,
                )
            ]

        for layer in layers:
            if isinstance(layer, TemporalBlock):
                layer.register_forward_hook(self.hook_fn)

        # Create the sequential model from the layers
        self.network = nn.Sequential(*layers)

    def hook_fn(self, module, input, output):
        """
        Hook function to store the activation maps

        Args:
            module: Module
            input: Input
            output: Output
        """
        self.activation_maps.append(output)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input data
        """
        return self.network(x)

    def get_activation_maps(self):
        """
        Get the activation maps
        """
        return self.activation_maps

    def clear_activation_maps(self):
        """
        Clear the activation maps
        """
        self.activation_maps.clear()
