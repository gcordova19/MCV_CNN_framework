import numpy as np
import random
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ToTensor(object):
    def __call__(self, image):
        #image = torch.from_numpy(np.flip(image.transpose((2, 0, 1)),axis=0).copy())
        image = torch.from_numpy(image.transpose((2, 0, 1)).copy())
        image = image.float()#.div(255)
        return image

'''class rescale(object):
    def __init__(self,rescale):
        self.rescale = rescale
    def __call__(self, image):
        return torch.mul(image, self.rescale)

class mean_norm(object):
    def __init__(self, mean):
        self.mean = torch.FloatTensor(mean)
    def __call__(self, image):
        return torch.add(image, torch.neg(self.mean))
        
class std_norm(object):
    def __init__(self, std):
        self.std = torch.FloatTensor(std)
    def __call__(self, image):
        return torch.div(image,(self.std + 1e-7))'''

class PrintInput(object):
    def __call__(self, image):
        print image
        input()


class rescale(object):
    def __init__(self,rescale):
        self.rescale = rescale
    def __call__(self, image):
        return image*self.rescale

class mean_norm(object):
    def __init__(self, mean):
        self.mean = np.asarray(mean, dtype=np.float32)
    def __call__(self, image):
        return image-self.mean
        
class std_norm(object):
    def __init__(self, std):
        self.std = np.asarray(std, dtype=np.float32)
    def __call__(self, image):
        return image/(self.std + 1e-7)
        
class preproces_input(object):
    def __init__(self, cf):
        self.cf = cf
        if self.cf.rescale is not None:
            self.rescale = rescale(self.cf.rescale)
        if self.cf.mean is not None:
            self.mean = mean_norm(self.cf.mean)
        if self.cf.std is not None:
            self.std = std_norm(self.cf.std)
    def __call__(self, image):
        if self.cf.rescale is not None:
            image = self.rescale(image)
        if self.cf.mean is not None:
            image = self.mean(image)
        if self.cf.std is not None:
            image = self.std(image)
        return image

class applyCrop(object):
    def __init__(self, cf):
        self.cf = cf
    def __call__(self, img, mask):
        h, w = self.cf.size_image_train
        th, tw = self.cf.crop_train
        if w == tw and h == th:
            return img, mask
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            img = img[y1:y1 + th,x1:x1 + tw]
            mask = mask[y1:y1 + th,x1:x1 + tw]
            '''img = img.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))'''
        return img, mask

class RandomHorizontalFlip(object):
    def __init__(self, cf):
        self.cf = cf
    def __call__(self, img, gt):
        if self.cf.hflips and random.random() < 0.5:
            return np.fliplr(img), np.fliplr(gt)
        return img, gt
