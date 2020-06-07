import scipy.io as io
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

class LSPDataset(data.Dataset):
    
    def __init__(self, path, image_size = 128, hmap_size = 0):
        self.image_size = image_size
        self.hmap_size = hmap_size
        self.path = path
        self.get_paths()
        self.joints = io.loadmat(self.path + 'joints.mat')['joints'].T
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
    def get_paths(self):
        self.paths = os.listdir(self.path + 'images/')
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = Image.open(self.path + 'images/' + self.paths[index])
        
        size = image.size
        x_scale = self.image_size/size[0]
        y_scale = self.image_size/size[1]
        image = self.transform(image)
        
        xjoints = (self.joints[index][:,:1] * x_scale)/self.image_size
        yjoints = (self.joints[index][:,1:2] * y_scale)/self.image_size
        joints = np.concatenate((xjoints, yjoints), axis=1).reshape(-1)

        mask = torch.tensor([m == 0 for m in self.joints[index][:,2:].reshape(-1)])

        center = sum(xjoints[xjoints>0])/len(xjoints[xjoints>0]), sum(yjoints[yjoints>0])/len(yjoints[yjoints>0])

        
        if self.target_type == 'points':
            return image, torch.from_numpy(joints).float(), center, mask

        if self.target_type == 'heatmap':
            points = joints.reshape(14, 2) * self.hmap_size
            hmap = gen_hmap(self.hmap_size, self.hmap_size, points, s=1.5)
            center = gen_hmap(self.hmap_size, self.hmap_size, [[center[0] * self.hmap_size, center[1] * self.hmap_size]], s=1.5)
            return image, torch.from_numpy(hmap).float(), center, mask

        
class LSPExtendedDataset(data.Dataset):
    
    def __init__(self, path, image_size = 128, hmap_size = 0):
        self.path = path
        self.get_paths()
        self.joints = io.loadmat(self.path + 'joints.mat')['joints'].T
        self.image_size = image_size
        self.hmap_size = hmap_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
    def get_paths(self):
        self.paths = os.listdir(self.path + 'images/')
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        
        image = Image.open(self.path + 'images/' + self.paths[index])
        size = image.size
        x_scale = self.image_size/size[0]
        y_scale = self.image_size/size[1]
        image = self.transform(image)
        
        xjoints = (self.joints[index].T[:,:1] * x_scale)/self.image_size
        yjoints = (self.joints[index].T[:,1:2] * y_scale)/self.image_size
        joints = np.concatenate((xjoints, yjoints), axis=1).reshape(-1)

        mask = self.joints[index].T[:,2:]
        mask = torch.from_numpy(mask).float().reshape(-1)

        center = sum(xjoints[xjoints>0])/len(xjoints[xjoints>0]), sum(yjoints[yjoints>0])/len(yjoints[yjoints>0])

        
        if self.hmap_size == 0:
            return image, torch.from_numpy(joints).float(), center, mask
        
        if self.hmap_size > 0:
            points = joints.reshape(14, 2) * self.hmap_size
            hmap = gen_hmap(self.hmap_size, self.hmap_size, points, s=1.5)
            center = gen_hmap(self.hmap_size, self.hmap_size, [[center[0] * self.hmap_size, center[1] * self.hmap_size]], s=1.5)
            return image, torch.from_numpy(hmap).float(), center, mask

def gaussian(x0, y0, sigma, width, height):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    return np.exp(-(((x-x0)**2 + (y - y0)**2 )/ (2 * sigma**2)))

def gen_hmap(height, width, points, s = 3):
    npoints = len(points)
    hmap = np.zeros((height, width, npoints), dtype = np.float)
    for i in range(npoints):
        if points[i][1] >= 0 and points[i][0] >= 0:
            if points[i][1] >= height: points[i][1] = height - 1
            if points[i][1] >= width: points[i][1] = width - 1
            hmap[:,:,i] = gaussian(points[i][1],points[i][0],s,height, width)
        else:
            hmap[:,:,i] = np.zeros((height, width))
    return hmap.T
        