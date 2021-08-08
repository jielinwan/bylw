import os
import numpy as np
from PIL import Image, ImageDraw
import h5py
import skimage
import torch
import torch.utils.data as data
from utils import write_log, _normalize, _denormalize
CK_ROOT = '../datasets/paper/'

class CKDataset(data.Dataset):

    def __init__(self, root=CK_ROOT, mode='train', use_hmaps=False, use_pmaps=False,
                 size_lr=(32, 32), size_hr=(128, 128), size_maps=(64, 64)):
        self.root = root
        self.mode = mode
        self.use_hmaps = use_hmaps
        self.use_pmaps = use_pmaps
        self.size_lr = size_lr
        self.size_hr = size_hr
        self.size_maps = size_maps
        self.img_info = list()

        self.data = []
        self.img_info = os.listdir(CK_ROOT)



    def __getitem__(self, index):

        # it will be returned
        image_lr = None
        hmaps = np.array([-1])
        pmaps = np.array([-1])

        # load infos
        face_name = self.img_info[index]
        face_img = Image.open(os.path.join(self.root, str(face_name))).convert('RGB')

        # resize
        image_lr = face_img.resize(self.size_lr, Image.BICUBIC)
        image_lr = image_lr.resize(self.size_hr, Image.BICUBIC)
        image_hr = face_img.resize(self.size_hr, Image.BICUBIC)



        return np.array(image_lr), np.array(image_hr),hmaps,pmaps

    def __len__(self):
        return len(self.img_info)



