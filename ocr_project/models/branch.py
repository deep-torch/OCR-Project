import torch.nn as nn


class Branch(nn.Module):
    def __init__(self, channels, strides):
        super(Branch, self).__init__()

        self.conv_layers = nn.Sequential()

        for i in range(1, len(channels)):
            self.conv_layers.add_module(f'conv_{i}', nn.Conv2d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=3, stride=strides[i-1], padding=(0, 1)))
            self.conv_layers.add_module(f'bn_{i}', nn.BatchNorm2d(channels[i]))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
            self.conv_layers.add_module(f'conv2_{i}', nn.Conv2d(in_channels=channels[i], out_channels=channels[i], kernel_size=3, padding='same'))
            self.conv_layers.add_module(f'bn2_{i}', nn.BatchNorm2d(channels[i]))
            self.conv_layers.add_module(f'relu2_{i}', nn.ReLU())

        self.projector = nn.LazyLinear(256)

    def forward(self, x):
        x = self.conv_layers(x)

        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3)).permute(0, 2, 1)
        x = self.projector(x)
        return x
