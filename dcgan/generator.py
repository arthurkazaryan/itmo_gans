from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as functional

from settings import GEN_FEATURES, CHANNELS, VECTOR_SIZE


class GeneratorTypes(str, Enum):
    csp_gan = "CSP"
    res_net_gan = "ResNet"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class CSPup(nn.Module):
    def __init__(self, conv_dims):
        super(CSPup, self).__init__()
        self.conv_dims = conv_dims

        self.left_1_transpose = nn.ConvTranspose2d(conv_dims // 2, conv_dims // 2, 4, 2, 1, bias=False)

        self.right_1_conv = nn.Conv2d(conv_dims // 2, conv_dims // 2, 1)
        # self.right_2_relu
        self.right_3_transpose = nn.ConvTranspose2d(conv_dims // 2, conv_dims // 2, 4, 2, 1, bias=False)
        self.right_4_conv = nn.Conv2d(conv_dims // 2, conv_dims // 2, 3, padding='same')
        # self.right_5_relu
        self.right_6_conv = nn.Conv2d(conv_dims // 2, conv_dims // 2, 3, padding='same')

    def forward(self, z):
        left = z[:, :self.conv_dims // 2, :, :]
        left = self.left_1_transpose(left)

        right = z[:, self.conv_dims // 2:, :, :]
        right = self.right_1_conv(right)
        right = functional.relu(right)
        right = self.right_3_transpose(right)
        right = self.right_4_conv(right)
        right = functional.relu(right)
        right = self.right_6_conv(right)

        z = torch.add(left, right)

        return z


class CSPGenerator(nn.Module):
    def __init__(self):
        super(CSPGenerator, self).__init__()

        self.layer_1_transpose = nn.ConvTranspose2d(VECTOR_SIZE, GEN_FEATURES * 8, 4, 1, 0, bias=False)
        self.layer_2_batchnorm = nn.BatchNorm2d(GEN_FEATURES * 8)
        self.layer_3_csp_up = CSPup(GEN_FEATURES * 8)
        self.layer_4_csp_up = CSPup(GEN_FEATURES * 4)
        self.layer_5_csp_up = CSPup(GEN_FEATURES * 2)
        self.layer_6_csp_up = CSPup(GEN_FEATURES)
        self.layer_7_conv = nn.Conv2d(GEN_FEATURES // 2, CHANNELS, 3, 1, 1, bias=False)

    def forward(self, array):
        array = self.layer_1_transpose(array)
        array = functional.relu(self.layer_2_batchnorm(array))
        array = self.layer_3_csp_up(array)
        array = self.layer_4_csp_up(array)
        array = self.layer_5_csp_up(array)
        array = self.layer_6_csp_up(array)
        array = torch.tanh(self.layer_7_conv(array))

        return array


class ResidualBlock(nn.Module):
    def __init__(self, conv_dims):
        super(ResidualBlock, self).__init__()
        self.conv_dims = conv_dims

        self.block_1_conv = nn.Conv2d(conv_dims, conv_dims, kernel_size=3, padding="same")
        self.block_1_bn = nn.BatchNorm2d(conv_dims)

        self.block_2_conv = nn.Conv2d(conv_dims, conv_dims, kernel_size=3, padding="same")
        self.block_2_bn = nn.BatchNorm2d(conv_dims)

        self.upsample = nn.ConvTranspose2d(conv_dims, conv_dims // 2, 4, 2, 1, bias=False)

    def forward(self, x):
        residual = x
        out = self.block_1_conv(x)
        out = self.block_1_bn(out)
        out = functional.relu(out)

        out = self.block_2_conv(out)
        out = self.block_2_bn(out)
        out = functional.leaky_relu(out)

        out += residual
        out = functional.relu(out)
        out = self.upsample(out)
        return out


class ResNetGenerator(nn.Module):
    def __init__(self):
        super(ResNetGenerator, self).__init__()

        self.layer_1_transpose = nn.ConvTranspose2d(VECTOR_SIZE, GEN_FEATURES * 8, 4, 1, 0, bias=False)
        self.layer_2_batchnorm = nn.BatchNorm2d(GEN_FEATURES * 8)
        self.layer_3_res_block = ResidualBlock(GEN_FEATURES * 8)
        self.layer_4_res_block = ResidualBlock(GEN_FEATURES * 4)
        self.layer_5_res_block = ResidualBlock(GEN_FEATURES * 2)
        self.layer_6_res_block = ResidualBlock(GEN_FEATURES)
        self.layer_7_conv = nn.Conv2d(GEN_FEATURES // 2, CHANNELS, 3, 1, 1, bias=False)

    def forward(self, array):
        array = self.layer_1_transpose(array)
        array = nn.ReLU()(self.layer_2_batchnorm(array))
        array = self.layer_3_res_block(array)
        array = self.layer_4_res_block(array)
        array = self.layer_5_res_block(array)
        array = self.layer_6_res_block(array)
        array = torch.tanh(self.layer_7_conv(array))

        return array
