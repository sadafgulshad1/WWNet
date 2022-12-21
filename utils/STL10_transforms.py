
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import os
import argparse
import numpy as np
from models import *

from torch.utils.data import Dataset
from torchvision import datasets
from collections import defaultdict, deque
import itertools
import random
random.seed(50)
import cv2
from torchvision.utils import save_image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
"""
Some transforms are taken from
@article{hendrycks2019robustness,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  journal={Proceedings of the International Conference on Learning Representations},
  year={2019}
}
"""



#from IPython.display import display
class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor
def generate_random_lines(imshape,slant,drop_length):
    drops=[]
    for i in range(6): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops
class rotate_scale(object):
    def __init__(self,angle=6,zoom_factor=0.757):
        self.angle = angle
        self.zoom_factor =zoom_factor
    def __call__(self, image):
        image = np.asarray(image)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        img = result
        if self.zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(img.shape[0] * self.zoom_factor))
            zw = int(np.round(img.shape[1] * self.zoom_factor))
            top = (img.shape[0] - zh) // 2
            left = (img.shape[1] - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = scizoom(img, (self.zoom_factor, self.zoom_factor, 1))
            img = out
        if self.zoom_factor>1:
            # clipping along the width dimension:
            ch0 = int(np.ceil(img.shape[0] / float(self.zoom_factor)))
            top0 = (img.shape[0] - ch0) // 2

            # clipping along the height dimension:
            ch1 = int(np.ceil(img.shape[1] / float(self.zoom_factor)))
            top1 = (img.shape[1] - ch1) // 2

            img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                          (self.zoom_factor, self.zoom_factor, 1), order=1)
        return img
class add_snow(object):
    def __call__(self, image):
        image = np.asarray(image)
        imshape = image.shape
        slant_extreme=8
        slant= np.random.randint(-slant_extreme,slant_extreme)
        drop_length=3
        drop_width=3
        drop_color=(0,0,0) ## a shade of gray
        rain_drops= generate_random_lines(imshape,slant,drop_length)
        for rain_drop in rain_drops:
            cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
        image= cv2.blur(image,(1,1)) ## rainy view are blurry
        brightness_coefficient = 0.7 ## rainy days are usually shady
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
        image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
        return image_RGB
class Wave_transform(object):
    # def__init__(self):
    def __call__(self,sample):
        image = np.asarray(sample)
        A = image.shape[0] / 3.0
        w = 2.0 / image.shape[1]
        sigma=random.uniform(0,0.19)
        shift = lambda x: A * np.sin(sigma*np.pi*x * w)

        for i in range(image.shape[0]):

            img_copy = image.copy()
            img_copy[:,i,:] = np.roll(img_copy[:,i,:], int(shift(i)),axis=0)
            image=img_copy
        return(image)

class Occlusion(object):
    def __init__(self,x=96,y=96):
        self.x=x
        self.y=y
        self.thickness=-1
        self.radius=25
    def __call__(self,sample):
        c_x= random.randint(self.radius,self.x)
        c_y= random.randint(self.radius,self.y)
        image=np.asarray(sample)
        h, w, _ = image.shape
        # h= random.randint(0,h)
        # w= random.randint(0,w)

        # out_image = cv2.rectangle(image, (0,0), (int(w/2) , h), (0, 0, 0), cv2.FILLED) #49%
        out_image = cv2.circle(image, (c_x,c_y), int(self.radius), (0, 0, 0), self.thickness)
        return (out_image)



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std=random.uniform(0,0.073)
        # self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class gaussian_blur(object):
    def __call__(self,sample):
        image=np.asarray(sample)
        sigma=random.uniform(0,0.95)
        # return(gaussian_filter(image, sigma=0.58))
        return(gaussian_filter(image, sigma=sigma))
class Motion_blur(object):

    def __init__(self, size=2):
        # assert isinstance(output_size, (int, tuple))
        size=random.randint(1,2.81)
        self.size = size
#
    def __call__(self, sample):
        # generating the kernel
        kernel_motion_blur = np.zeros((self.size, self.size))
        kernel_motion_blur[int((self.size-1)/2), :] = np.ones(self.size)
        kernel_motion_blur = kernel_motion_blur / self.size

        # applying the kernel to the input image
        image=np.asarray(sample)


        output = cv2.filter2D(image, -1, kernel_motion_blur)
        # output = cv2.filter2D(output, -1, kernel_motion_blur)

        return (output)
# Function to distort image
class elastic_transform_class(object):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    def __init__(self, alpha=2.56, sigma=1.28, alpha_affine=1.6, random_state=None):
        self.random_state=random_state
        if self.random_state is None:
            self.random_state = np.random.RandomState(None)
        self.alpha=alpha
        self.sigma=sigma
        self.alpha_affine=alpha_affine

    def __call__(self,sample):
        image = np.asarray(sample)
        a=random.uniform(0.06,0.16)
        a=a*32
        shape = image.shape
        shape_size = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + self.random_state.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma) * a
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma) * a
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

class WarpAffine(object):

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        x = np.asarray(x)
        img_size = x.shape[:2]
        frame_center = np.float32(img_size) // 2
        frame_size = min(img_size) // 3
        pts1 = np.float32([frame_center + frame_size,
                           [frame_center[0] + frame_size, frame_center[1] - frame_size],
                           frame_center - frame_size])
        pts2 = pts1 + np.random.uniform(-self.alpha * min(img_size),
                                        self.alpha * min(img_size), size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(x, M, img_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    def __repr__(self):
        return self.__class__.__name__ + "(alpha={.3f})".format(self.alpha)


def elastic_transform(img, dx, dy):
    if len(img.shape) == 3:
        x, y, z = np.meshgrid(np.arange(img.shape[1]), np.arange(
            img.shape[0]), np.arange(img.shape[2]))
        ind_x = np.reshape(y + dy, (-1, 1))
        ind_y = np.reshape(x + dx, (-1, 1))
        ind_z = np.reshape(z, (-1, 1))
        return map_coordinates(img, (ind_x, ind_y, ind_z), order=1, mode='reflect').reshape(img.shape)
    else:
        x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        ind_x = np.reshape(y + dy, (-1, 1))
        ind_y = np.reshape(x + dx, (-1, 1))
        return map_coordinates(img, (ind_x, ind_y), order=1, mode='reflect').reshape(img.shape)


class Elastic(object):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    def __init__(self, alpha=0.25, sigma=0.07):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        image = np.asarray(sample)
        shape = image.shape

        sigma = min(shape) * self.sigma
        alpha = min(shape) * self.alpha

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        return elastic_transform(image, dx, dy)

    def __repr__(self):
        str_ = self.__class__.__name__
        str_ = str_ + ("(alpha={alpha:.2f}, sigma={sigma:.2f})".format(**self.__dict__))
        return str_


class ElasticSeparable(object):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.comx/erniejunior/601cdf56d2b424757de5
    """

    def __init__(self, alpha=0.19, sigma=0.07):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        image = np.asarray(sample)
        shape = image.shape

        sigma = min(shape) * self.sigma
        alpha = min(shape) * self.alpha

        dx = np.random.rand(*shape)
        dy = np.random.rand(*shape)
        dx = dx[[0]].repeat(shape[0], 0)
        dy = dy[:, [0]].repeat(shape[1], 1)

        dx = gaussian_filter(dx * 2 - 1, sigma) * alpha
        dy = gaussian_filter(dy * 2 - 1, sigma) * alpha

        return elastic_transform(image, dx, dy)

    def __repr__(self):
        str_ = self.__class__.__name__
        str_ = str_ + ("(alpha={alpha:.2f}, sigma={sigma:.2f})".format(**self.__dict__))
        return str_
class ElasticParSep(object):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.comx/erniejunior/601cdf56d2b424757de5
    """

    def __init__(self, alpha=0.275, sigma=0.07):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        image = np.asarray(sample)
        shape = image.shape

        sigma = min(shape) * self.sigma
        alpha = min(shape) * self.alpha

        # dx = np.random.rand(*shape)
        dy = np.random.rand(*shape)
        # dx = dx[[0]].repeat(shape[0], 0)
        dy = dy[:, [0]].repeat(shape[1], 1)

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter(dy * 2 - 1, sigma) * alpha

        return elastic_transform(image, dx, dy)

    def __repr__(self):
        str_ = self.__class__.__name__
        str_ = str_ + ("(alpha={alpha:.2f}, sigma={sigma:.2f})".format(**self.__dict__))
        return str_

def clipped_zoom(img, zoom_factor):
    # clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    # clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                  (zoom_factor, zoom_factor, 1), order=1)

    return img
from scipy.ndimage import zoom as scizoom
def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1

def gauss_function(x, mean, sigma):
    return (np.exp(- x**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z
def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted
def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred
class snow():
    ##code taken from https://github.com/bethgelab/imagecorruptions
    def __init__(self, severity=1):
        self.severity=severity
        self.c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][self.severity - 1]
    def __call__(self,sample):
        x=sample
        x = np.array(x, dtype=np.float32) / 255.
        snow_layer = np.random.normal(size=x.shape[:2], loc=self.c[0],
                                  scale=self.c[1])  # [:2] for monochrome

        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], self.c[2])
        snow_layer[snow_layer < self.c[3]] = 0

        snow_layer = np.clip(snow_layer.squeeze(), 0, 1)


        snow_layer = _motion_blur(snow_layer, radius=self.c[4], sigma=self.c[5], angle=np.random.uniform(-135, -45))

        # The snow layer is rounded and cropped to the img dims
        snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
        snow_layer = snow_layer[..., np.newaxis]
        snow_layer = snow_layer[:x.shape[0], :x.shape[1], :]

        if len(x.shape) < 3 or x.shape[2] < 3:
            x = self.c[6] * x + (1 - self.c[6]) * np.maximum(x, x.reshape(x.shape[0],
                                                                x.shape[
                                                                    1]) * 1.5 + 0.5)
            snow_layer = snow_layer.squeeze(-1)
        else:
            x = self.c[6] * x + (1 - self.c[6]) * np.maximum(x, cv2.cvtColor(x,
                                                                   cv2.COLOR_RGB2GRAY).reshape(
                x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
        try:
            return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        except ValueError:
            print('ValueError for Snow, Exception handling')
            x[:snow_layer.shape[0], :snow_layer.shape[1]] += snow_layer + np.rot90(
                snow_layer, k=2)
            return (np.clip(x, 0, 1) * 255)
