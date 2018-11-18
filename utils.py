from __future__ import division
import torch, cv2
import torch.nn as nn
import numpy as np
import scipy, scipy.misc
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import matrix_rank
from math import floor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob, os, random


def sample(_list, _num):
    rand_num = random.sample(range(0, len(_list)), _num)
    return [_list[rand_num[i]] for i in range(len(rand_num))]


def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball """
    # calculate the length of the vector, but numpy will output a row instead of a column
    # if we want to retain the dimension of batch_size, we will need to do [:, newaxis]
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)


def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)


def crop_full_chip(data_dir, dataname):
    img_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
    img = Image.open(img_paths[0]).convert('RGB')
    
    if dataname == 'flower_chip':
        stride = 512
        size = 512
    else:
        stride = 128
        size = 128
    [h, w] = img.size
    num_h = int(floor((w-size) // stride)+1)
    num_v = int(floor((h-size) // stride)+1)

    img_list = []
    for j in range(num_v):
        for i in range(num_h):
            start_x = i * stride
            start_y = j * stride
            box = (start_x, start_y, start_x+size, start_y+size)
            tmp_img = img.crop(box)
            img_list.append(tmp_img)
    return img_list, num_h, num_v
    
def load_full_chip(data_dir, dataname, batch_size, img_size=64, convert='RGB'):
    ''' load data for full chip '''
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    # crop the image first
    img_list, num_h, num_v = crop_full_chip(data_dir, dataname)
    
    dataset = data_folder(img_list, transform, convert)
    # we retain the position of original chip
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader, num_h, num_v

def load_multi(data_dir, batch_size, img_size=128, convert='RGB'):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    dataset = data_folder_multi(data_dir, transform, convert)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class data_folder_multi(Dataset):
    def __init__(self, data_dir, transform=None, convert='RGB'):
        self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        print(len(self.image_paths))
        self.image_paths = self.check_image_folder()
        print(len(self.image_paths))
        self.transform = transform
        self.convert = convert
    def check_image_folder(self):
        length = len(self.image_paths)
        #print()
        if length < 10:
            paths = []
            for i in range(length):
                folder = self.image_paths[i]
                tmp_paths = list(map(lambda x: os.path.join(folder,x), os.listdir(folder)))
                paths = paths + tmp_paths
            return paths 
        else:
            return 1

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert(self.convert)
        label = np.zeros((10), dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, index
    def __len__(self):
        return len(self.image_paths)

def load(data_dir, batch_size, img_size=64, convert='RGB'):
    ''' load data in general'''
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()]) # do not need to normalize
    dataset = data_folder(data_dir, transform, convert)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

class data_folder(Dataset):    
    ''' folder for general dataset '''
    def __init__(self, data_dir, transform=None, convert='RGB'):
        ''' initialize image paths and preprocessing module '''
        if type(data_dir) is list:
            self.flag = 1
            self.image_paths = data_dir
        else:
            self.flag = 0
            self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        self.transform = transform
        self.convert = convert

    def __getitem__(self, index):
        ''' reads an image from a file and preprocesses it and returns '''
        #print(index)
        if self.flag == 0:
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert(self.convert)
        else:
            image = self.image_paths[index]
        # we choose arbitrary labels because we do not use them
        label = np.zeros((10), dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, index
 
    def __len__(self):
        ''' return the total number of image files '''
        return len(self.image_paths)

def load_random(data_dir, batch_size, num, img_size=64, convert='RGB'):
    ''' load data in random sample'''
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()]) # do not need to normalize
    dataset = data_folder_random(data_dir, num, transform, convert)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

class data_folder_random(Dataset):    
    ''' folder for random sample dataset '''
    def __init__(self, data_dir, num, transform=None, convert='RGB'):
        ''' initialize image paths and preprocessing module '''
        if type(data_dir) is list:
            self.flag = 1
            self.image_paths = data_dir
        else:
            self.flag = 0
            self.image_paths = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
        self.transform = transform
        self.convert = convert

    def __getitem__(self, index):
        ''' reads an image from a file and preprocesses it and returns '''
        #print(index)
        if self.flag == 0:
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert(self.convert)
        else:
            image = self.image_paths[index]
        # we choose arbitrary labels because we do not use them
        label = np.zeros((10), dtype=np.float)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, index
 
    def __len__(self):
        ''' return the total number of image files '''
        return len(self.image_paths)


class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDL = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def loss_plot(loss_list, path):
    x = range(len(loss_list))
    y = loss_list

    plt.plot(x, y, label='loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    
def ip_z(z1, z2, batch_size):
    dim = z1.shape[1]
    z = torch.zeros((batch_size, dim))
    for i in range(batch_size):
        z[i] = torch.lerp(z1, z2, (i/batch_size))
    
    return z
