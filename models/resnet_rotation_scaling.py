import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from .layers.rot_scaling_conv import DistConv_H_H, DistConv_Z2_H, DistConv_H_H_1x1, project #rotation scaling
from .transfer_imagenet_weights import transfer_weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ': {}->{}'.format(self.in_planes, self.planes)


class DistBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_displacements=9, scale=1, alpha=1, **kwargs):
        super(DistBasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = DistConv_H_H(in_planes, planes, kernel_size=7, effective_size=3,
                                  num_displacements=num_displacements, stride=stride,
                                  padding=3, bias=False, scale=scale, alpha=alpha)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = DistConv_H_H(planes, planes, kernel_size=7, effective_size=3,
                                  num_displacements=num_displacements, stride=1,
                                  padding=3, bias=False, scale=scale, alpha=alpha)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                DistConv_H_H_1x1(in_planes, self.expansion * planes, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ': {}->{}'.format(self.in_planes, self.planes)


class ProjectionBasicBlock(BasicBlock):

    def forward(self, x):
        x = project(x)
        return super().forward(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(Bottleneck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ': {}->{}'.format(self.in_planes, self.planes)


class ProjectionBottleneck(Bottleneck):

    def forward(self, x):
        x = project(x)
        return super().forward(x)


class DistBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, num_displacements=9, scale=1, alpha=1, **kwargs):
        super(DistBottleneck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = DistConv_H_H_1x1(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = DistConv_H_H(planes, planes, kernel_size=7, effective_size=3,
                                  num_displacements=num_displacements, stride=stride,
                                  padding=3, bias=False, scale=scale, alpha=alpha)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = DistConv_H_H_1x1(planes, self.expansion * planes, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                DistConv_H_H_1x1(in_planes, self.expansion * planes, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ': {}->{}'.format(self.in_planes, self.planes)


class DistBottleneckProjection(DistBottleneck):

    def forward(self, x):
        return project(super().forward(x))


class ResNet(nn.Module):
    def __init__(self, blocks, num_blocks, num_classes=10, num_displacements=9, scale=1, alpha=1):
        assert len(blocks) == sum(num_blocks)
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = DistConv_Z2_H(3, 64, kernel_size=11, effective_size=7, num_displacements=num_displacements,
                                   scale=scale, alpha=alpha, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)

        blocks_1 = blocks[:num_blocks[0]]
        blocks = blocks[num_blocks[0]:]
        self.layer1 = self._make_layer(blocks_1, 64, num_blocks[0], stride=1,
                                       num_displacements=num_displacements,
                                       scale=scale, alpha=alpha)

        blocks_2 = blocks[:num_blocks[1]]
        blocks = blocks[num_blocks[1]:]
        self.layer2 = self._make_layer(blocks_2, 128, num_blocks[1], stride=2,
                                       num_displacements=num_displacements,
                                       scale=scale, alpha=alpha)

        blocks_3 = blocks[:num_blocks[2]]
        blocks_4 = blocks[num_blocks[2]:]
        self.layer3 = self._make_layer(blocks_3, 256, num_blocks[2], stride=2,
                                       num_displacements=num_displacements,
                                       scale=scale, alpha=alpha)
        self.layer4 = self._make_layer(blocks_4, 512, num_blocks[3], stride=2,
                                       num_displacements=num_displacements,
                                       scale=scale, alpha=alpha)
        self.fc = nn.Linear(512 * blocks[-1].expansion, num_classes)

    def _make_layer(self, blocks, planes, num_blocks, stride, num_displacements, scale, alpha):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block, stride in zip(blocks, strides):
            layers.append(block(self.in_planes, planes, stride,
                                num_displacements=num_displacements,
                                scale=scale, alpha=alpha))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.mean(-1).mean(-1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def _partial_elastic_resnet(block_type, num_elastic_blocks, num_blocks, **kwargs):
    if block_type == 'basic':
        regular_block = BasicBlock
        elastic_block = DistBasicBlock
        proj_block = ProjectionBasicBlock
    elif block_type == 'bottleneck':
        regular_block = Bottleneck
        elastic_block = DistBottleneck
        proj_block = ProjectionBottleneck
    else:
        raise KeyError

    num_regular_blocks = sum(num_blocks) - 1 - num_elastic_blocks
    blocks = [elastic_block] * num_elastic_blocks
    blocks = blocks + [proj_block] + [regular_block] * num_regular_blocks
    return ResNet(blocks, num_blocks, **kwargs)


def resnet_rotation_scaling_18(num_classes=10, num_elastic_blocks=0, basis_num_displacements=4, basis_scale=0.9, basis_alpha=1,**kwargs):
    model = _partial_elastic_resnet('basic', num_elastic_blocks=num_elastic_blocks,
                                    num_blocks=[2, 2, 2, 2], num_classes=num_classes,
                                    num_displacements=basis_num_displacements, scale=basis_scale, alpha=basis_alpha)
    model.load_state_dict(
      transfer_weights(tvm.resnet18(True).state_dict(),
                       model.state_dict()
                       ))
    return model
def resnet_rotation_scaling_34(num_classes=10, num_elastic_blocks=0, num_displacements=4, scale=0.9, alpha=1):
    model = _partial_elastic_resnet('basic', num_elastic_blocks=num_elastic_blocks,
                                    num_blocks=[3, 4, 6, 3], num_classes=num_classes,
                                    num_displacements=num_displacements, scale=scale, alpha=alpha)

    model.load_state_dict(
        transfer_weights(tvm.resnet34(True).state_dict(),
                         model.state_dict()
                         ))
    return model


def resnet_rotation_scaling_50(num_classes=10, num_elastic_blocks=0, num_displacements=4, scale=0.9, alpha=1):
    model = _partial_elastic_resnet('bottleneck', num_elastic_blocks=num_elastic_blocks,
                                    num_blocks=[3, 4, 6, 3], num_classes=num_classes,
                                    num_displacements=num_displacements, scale=scale, alpha=alpha)

    model.load_state_dict(
        transfer_weights(tvm.resnet50(True).state_dict(),
                         model.state_dict()
                         ))
    return model


def resnet_rotation_scaling_101(num_classes=10, num_elastic_blocks=0, num_displacements=4, scale=0.9, alpha=1):
    model = _partial_elastic_resnet('bottleneck', num_elastic_blocks=num_elastic_blocks,
                                    num_blocks=[3, 4, 23, 3], num_classes=num_classes,
                                    num_displacements=num_displacements, scale=scale, alpha=alpha)

    model.load_state_dict(
        transfer_weights(tvm.resnet101(True).state_dict(),
                         model.state_dict()
                         ))
    return model


def resnet_rotation_scaling_152(num_classes=10, num_elastic_blocks=0, basis_num_displacements=4, basis_scale=0.9, basis_alpha=1,**kwargs):
    model = _partial_elastic_resnet('bottleneck', num_elastic_blocks=num_elastic_blocks,
                                    num_blocks=[3, 8, 36, 3], num_classes=num_classes,
                                    num_displacements=basis_num_displacements, scale=basis_scale, alpha=basis_alpha)
    model.load_state_dict(
        transfer_weights(tvm.resnet152(True).state_dict(),
                         model.state_dict()
                         ))
    return model

