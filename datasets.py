import os
import torch
from PIL import Image
from scipy.io import loadmat


class FreyFaceDataset(torch.utils.data.Dataset):
    # data_file: available at https://cs.nyu.edu/~roweis/data/frey_rawface.mat
    data_file = 'frey_rawface.mat'

    def __init__(self, root, transform=None):
        super(FreyFaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        if not self._check_exists():
            raise RuntimeError('Dataset do not found in the directory \"{}\". \nYou can download FreyFace '
                               'dataset from https://cs.nyu.edu/~roweis/data/frey_rawface.mat '.format(self.root))
        self.data = loadmat(os.path.join(self.root, self.data_file))['ff'].T

    def __getitem__(self, index):
        img = self.data[index].reshape(28, 20)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.data_file))
