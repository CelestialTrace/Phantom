import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

momentum = 0.001

def mish(x):
    """Mish activation (unused by default)."""
    return x * torch.tanh(F.softplus(x))

class PSBatchNorm2d(nn.BatchNorm2d):
    """Optional variant; inherits torch BN."""
    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
    def forward(self, x):
        return super().forward(x) + self.alpha

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0,
                 activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=momentum, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=momentum, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=True
        ) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block,
                 stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(
                i == 0 and in_planes or out_planes,
                out_planes,
                i == 0 and stride or 1,
                drop_rate,
                activate_before_residual if i == 0 else False
            ))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28,
                 widen_factor=2, drop_rate=0.0, is_remix=False):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0), 'Depth must be 6n+4'
        n = (depth - 4) / 6

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], BasicBlock,
            first_stride, drop_rate, activate_before_residual=True
        )
        self.block2 = NetworkBlock(n, channels[1], channels[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], BasicBlock, 2, drop_rate)

        self.bn1 = nn.BatchNorm2d(channels[3], momentum=momentum, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(self.channels, 4)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).view(-1, self.channels)
        logits = self.fc(out)

        if ood_test:
            return logits, out
        if self.is_remix:
            rot = self.rot_classifier(out)
            return logits, rot
        return logits

class build_WideResNet:
    """Mirror TorchSSL’s builder in `utils.net_builder`."""
    def __init__(self, first_stride=1, depth=28, widen_factor=2,
                 bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.dropRate = dropRate
        self.is_remix = is_remix

    def build(self, num_classes):
        return WideResNet(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )
    

def main():
    device = torch.device("cuda")

    # Transforms for test set (using CIFAR-10 normalization statistics)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2
    )

    # print model architecture
    # wrn = build_WideResNet(1, depth=28, widen_factor=2).build(10)
    # print(wrn)
    # load the model
    model = build_WideResNet(first_stride=1, depth=28, widen_factor=2).build(10)
    ckpt = torch.load("./latest_model.pth", map_location=device)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "ema_model" in ckpt:
        state_dict = ckpt["ema_model"]    # if you want to evaluate the EMA model instead
    else:
        state_dict = ckpt                  # fallback if it really was just a raw state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Accuracy of the model on the CIFAR-10 test images: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
