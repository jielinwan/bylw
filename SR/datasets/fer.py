import os
import numpy as np
from PIL import Image, ImageDraw
import h5py
import torch
import torch.utils.data as data
from utils import write_log, _normalize, _denormalize
CELEB_ROOT = '../datasets/CelebA-HQ_ParsingMap/'
check_dir='../FSRGAN-OUT//output_test/'
filename=file_name = os.path.join(check_dir, '02.pth')
G = torch.load(file_name)
G.eval()

class FERDataset(data.Dataset):

    def __init__(self, root=CELEB_ROOT, mode='train', use_hmaps=False, use_pmaps=False,
                 size_lr=(48, 48), size_hr=(192, 192), size_maps=(96, 96)):
        self.root = root
        self.mode = mode
        self.use_hmaps = use_hmaps
        self.use_pmaps = use_pmaps
        self.size_lr = size_lr
        self.size_hr = size_hr
        self.size_maps = size_maps
        self.img_info = list()

        self.data = h5py.File('../datasets/data.h5', 'r', driver='core')
        self.train_data = self.data['Training_pixel']
        self.train_labels = self.data['Training_label']
        self.train_data = np.asarray(self.train_data)
        self.train_data = self.train_data.reshape((28709, 48, 48))



    def __getitem__(self, index):

        # it will be returned
        image_lr = None
        hmaps = np.array([-1])
        pmaps = np.array([-1])

        # load infos
       # face_idx = self.img_info[index]['image']
        #face_img = Image.open(os.path.join(self.root, 'CelebA-HQ-img', str(face_idx) + '.jpg')).convert('RGB')

        img, target = self.train_data[index], self.train_labels[index]
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

        #img = Image.fromarray(img)


        return img, hmaps, pmaps

    def __len__(self):
        return len(self.train_data)



