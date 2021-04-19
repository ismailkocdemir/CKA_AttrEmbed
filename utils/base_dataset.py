import copy
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms

import io

class BaseDataset(Dataset):
    def __init__(self,const):
        super(BaseDataset,self).__init__()
        self.const = copy.deepcopy(const)
        if self.const.download==True:
            io.mkdir_if_not_exists(self.const.root)

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

        label = self.labels[idx]

        to_return = {
            'img': np.array(img),
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
