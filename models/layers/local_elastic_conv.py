import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
def hermite_poly(X, n):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array

    Output:
        Y: array of shape X.shape

    -------------------------
    BSD 3-Clause "New" or "Revised" License License
    Copyright (c) 2005-2020, NumPy Developers.
    Copyright (c) 2020, Ivan Sosnovik

    -------------------------
    Source: https://github.com/numpy/numpy/blob/master/numpy/polynomial/hermite_e.py
    """
    c = torch.Tensor([0] * n + [1]).to(X.device)
    c = c.reshape(c.shape + (1,) * X.dim())

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * X
    return c0 + c1 * X


def get_basis_for_grid(X, Y, effective_size, scale=0.9):
    """
    Args:
        X, Y: 2D grid, e.g.
            X = tensor([[-2., -1.,  0.,  1.,  2.],
                        [-2., -1.,  0.,  1.,  2.],
                        [-2., -1.,  0.,  1.,  2.],
                        [-2., -1.,  0.,  1.,  2.],
                        [-2., -1.,  0.,  1.,  2.]])

            Y = tensor([[-2., -2., -2., -2., -2.],
                        [-1., -1., -1., -1., -1.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 1.,  1.,  1.,  1.,  1.],
                        [ 2.,  2.,  2.,  2.,  2.]])

        effective_size: number of filters = effective_size**2
        scale: spatial parameter of the basis

    Output:
        basis: tensor of shape [effective_size**2, X.shape[0], X.shape[1]]

    """
    G = torch.exp(-(X**2 + Y**2) / (2 * scale**2))

    basis = []
    for ny in range(effective_size):
        for nx in range(effective_size):
            basis.append(G * hermite_poly(X / scale, nx) * hermite_poly(Y / scale, ny))

    return torch.stack(basis)

def elastic_transform_local(X,Y, alpha, sigma,size,n):

    x, y = np.meshgrid(np.arange(size), np.arange(size))
    XY_array = [(np.asarray(X),np.asarray(Y))]
    XY_tensor = [(X,Y)]
    for i in range (n):
        
        #### if don't want to use affine comment this sec
        random_state = np.random.RandomState(None)
        shape_size = (size,size)
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        XY_array_0 = cv2.warpAffine(np.asarray(XY_array[0][0]), M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        XY_array_1 = cv2.warpAffine(np.asarray(XY_array[0][1]), M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        ####
        dx = gaussian_filter((np.random.rand(size) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(size) * 2 - 1), sigma) * alpha
        ind_x = np.reshape(y + dy, (-1, 1))
        ind_y = np.reshape(x + dx, (-1, 1))
        A = torch.tensor(map_coordinates(XY_array_0, (ind_x, ind_y), order=1, mode='reflect').reshape((size,size)))
        B = torch.tensor(map_coordinates(XY_array_1, (ind_x, ind_y), order=1, mode='reflect').reshape((size,size)))

        XY_tensor.append((A, B))
    return XY_tensor


def Elastic_local(size, effective_size,n,  alpha=1, scale=0.9):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """


    alpha = alpha
    sigma = scale
    basis = []
    X = torch.linspace(-(size // 2), size // 2, size)
    Y = torch.linspace(-(size // 2), size // 2, size)
    X = X[None, :].repeat(size, 1)
    Y = Y[:, None].repeat(1, size)

    grids = elastic_transform_local(X,Y, alpha, sigma, size, n)

    for X, Y in grids:
        basis.append(get_basis_for_grid(X, Y, effective_size, scale=scale))

    return torch.stack(basis, 1)


class ElasticBasis(nn.Module):

    def __init__(self, size, effective_size, num_displacements, scale=0.9, alpha=1):
        super().__init__()
        self.size = size
        self.effective_size = effective_size
        self.num_funcs = effective_size**2
        
        self.num_displacements = num_displacements
        self.num_elements = self.num_displacements + 1
        self.scale = scale
        self.alpha = alpha
        basis = Elastic_local(size=self.size, effective_size=self.effective_size,n=self.num_displacements,
                               alpha=self.alpha, scale=self.scale)
        
        norm = basis.pow(2).sum(-1).sum(-1).sqrt()[:, 0][:, None, None, None]
        basis = basis / norm
        self.register_buffer('basis', basis)

    def forward(self, weight):
        
        kernel = weight @ self.basis.view(self.num_funcs, -1)


        kernel = kernel.view(*weight.shape[:-1], self.num_elements, self.size, self.size)
        return kernel

    def extra_repr(self):
        s = '{size}x{size} | num_elements={num_elements} | alpha={alpha} | scale={scale} | num_funcs={num_funcs}'
        return s.format(**self.__dict__)


class DistConv_Z2_H(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 num_displacements=8, stride=1, padding=0, bias=False, scale=0.9, alpha=1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.num_displacements = num_displacements
        self.stride = stride
        self.padding = padding
        self.scale = scale
        self.alpha = alpha
        self.basis = ElasticBasis(kernel_size, effective_size,
                                  num_displacements, scale=scale, alpha=alpha)
        assert self.basis.size == self.kernel_size

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.basis.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        kernel = self.basis(self.weight)

        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        
        # convolution
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, -1, H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels}'
        return s.format(**self.__dict__)


class DistConv_H_H(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 num_displacements=8, stride=1, padding=0, bias=False, scale=0.9, alpha=1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.num_displacements = num_displacements
        self.stride = stride
        self.padding = padding
        self.scale = scale
        self.alpha = alpha
        self.basis = ElasticBasis(kernel_size, effective_size,
                                  num_displacements, scale=scale, alpha=alpha)
        assert self.basis.size == self.kernel_size

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.basis.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        kernel = self.basis(self.weight)

        # expand kernel
        kernel = kernel.permute(2, 0, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        B, C, S, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, -1, H, W)
        output = F.conv2d(x, kernel, padding=self.padding, groups=S, stride=self.stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}->{out_channels}'
        return s.format(**self.__dict__)


class DistConv_H_H_1x1(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = (1, stride, stride)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1, 1))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return F.conv3d(x, self.weight, stride=self.stride)

    def extra_repr(self):
        s = '{in_channels}->{out_channels}'
        return s.format(**self.__dict__)


class Projection(nn.Module):

    def forward(self, x):
        return x.max(2)[0]


def project(x):
    return x.max(2)[0]
