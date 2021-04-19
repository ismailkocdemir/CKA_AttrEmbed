import os
import copy
#import itertools
import pickle
import numpy as np
#from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
import scipy

from utils import io

def get_dataset(const):
    if const.dataset_type == "Cifar100":
        return Cifar100Dataset(const)
    elif const.dataset_type == 'STL10':
        return STL10Dataset(const)
    elif const.dataset_type == 'VOCDetection':
        return VOCDetectionDataset(const)
    else:
        raise NotImplementedError("{} is not implemented yet".format(dataset_type))

class DatasetConstants(io.JsonSerializableClass):
    def __init__(self, root, download, train):
        super(DatasetConstants,self).__init__()
        self.root = root
        self.download = download
        self.train = train

class BaseDataset(Dataset, ABC):
    def __init__(self,const):
        super(BaseDataset,self).__init__()
        self.const = copy.deepcopy(const)
        if self.const.download==True:
            io.mkdir_if_not_exists(self.const.root)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

    @abstractmethod
    def load_labels(self):
        '''fill in acccording to the chosen dataset. used by <__getitem__>'''
        pass
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,i):
        img,idx = self.dataset[i]
        if self.const.train==True:
            img = self.transforms(img)
        else:
            img = self.transforms_test(img)

        label = self.labels[idx]

        to_return = {
            'img': img,
            'label_idx': idx,
            'label': label
        }
        return to_return

    def normalize(self,imgs,mean,std):
        imgs = (imgs-mean) / std
        return imgs

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = [s for s in batch if s is not None]
            return default_collate(batch)

        return collate_fn

class VOCDetectionDataset(BaseDataset):
    def __init__(self,const):
        super().__init__(const)
        self.dataset = torchvision.datasets.STL10(
            self.const.root,
            split='train' if self.const.train else 'test',
            download=self.const.download
        )
        self.labels = self.load_labels()
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96,padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

    def load_labels(self):
        meta_file = os.path.join(
            self.const.root,
            'stl10_binary/class_names.txt')
        with open(meta_file,'r') as mf:
            labels = mf.readlines()
            labels = np.array([lb.strip() for lb in labels])
        return labels


class STL10Dataset(BaseDataset):
    def __init__(self,const):
        super().__init__(const)
        self.dataset = torchvision.datasets.STL10(
            self.const.root,
            split='train' if self.const.train else 'test',
            download=self.const.download
        )
        self.labels = self.load_labels()
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96,padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

    def load_labels(self):
        meta_file = os.path.join(
            self.const.root,
            'stl10_binary/class_names.txt')
        with open(meta_file,'r') as mf:
            labels = mf.readlines()
            labels = np.array([lb.strip() for lb in labels])
        return labels

class Cifar100Dataset(BaseDataset):
    def __init__(self,const):
        super().__init__(const)
        self.dataset = torchvision.datasets.CIFAR100(
            self.const.root,
            self.const.train,
            download=self.const.download)
        self.labels = self.load_labels()
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

        
    def load_labels(self):
        meta_file = os.path.join(
            self.const.root,
            'cifar-100-python/meta')
        fo = open(meta_file,'rb')
        labels = pickle.load(fo,encoding='latin1')['fine_label_names']
        return labels

if __name__=='__main__':
    const = DatasetConstants()
    const.download = False
    dataset = STL10Dataset(const)
    import pdb; pdb.set_trace()
    import scipy
    outdir = os.path.join(
        os.getcwd(),
        'exp/scratch')
    io.mkdir_if_not_exists(outdir)
    for i in range(10):
        data = dataset[i]
        img = data['img']
        label = data['label']
        filename = os.path.join(outdir,f'{i}_{label}.png')
        scipy.misc.imsave(filename,img)

    # dataloader = DataLoader(dataset,batch_size=2)
    # for data in dataloader:
    #     import pdb; pdb.set_trace()
