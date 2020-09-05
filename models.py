from torch.nn import Module, Conv2d
import torch.nn.functional as F
import resnet


class SteeringCommandsDQN(Module):
    def __init__(self, num_input_channels=3, num_output_channels=4):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels, num_classes=num_output_channels)

    def forward(self, x):
        return self.resnet18(x)

class DenseActionSpaceDQN(Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv2 = Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv3 = Conv2d(32, num_output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.resnet18.features(x)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(x)
