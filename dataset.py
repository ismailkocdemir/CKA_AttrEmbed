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

import scipy

from utils import io

class Cifar100DatasetConstants(io.JsonSerializableClass):
    def __init__(
            self,
            root=os.path.join(os.getcwd(),'data/cifar100')):
        super(Cifar100DatasetConstants,self).__init__()
        self.root = root
        self.download = False
        self.train = True
        #self.num_held_out_classes = 20 # 20, 40, 60, 80

class Cifar100Dataset(Dataset):
    def __init__(self,const):
        super(Cifar100Dataset,self).__init__()
        self.const = copy.deepcopy(const)
        if self.const.download==True:
            io.mkdir_if_not_exists(self.const.root)
        self.dataset = torchvision.datasets.CIFAR100(
            self.const.root,
            self.const.train,
            download=self.const.download)
        self.labels = self.load_labels()
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=2),
        ])

    def load_labels(self):
        meta_file = os.path.join(
            self.const.root,
            'cifar-100-python/meta')
        fo = open(meta_file,'rb')
        labels = pickle.load(fo,encoding='latin1')['fine_label_names']
        return labels
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,i):
        img,idx = self.dataset[i]
        if self.const.train==True:
            img = self.transforms(img)

        label = self.labels[idx]

        to_return = {
            'img': np.array(img),
            'label_idx': idx,
            'label': label
            #'super_label': self.fine_to_super[label],
            #'super_label_idx': self.fine_idx_to_super_idx[idx],
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


if __name__=='__main__':
    const = Cifar100DatasetConstants()
    const.download = False
    dataset = Cifar100Dataset(const)
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
