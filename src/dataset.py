from glob import glob
from PIL import Image
import random
import os
from torch.utils.data import Dataset


class Places2(Dataset):
    def __init__(self, data_root, img_transform, mask_transform, data='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.data_root = data_root

        # get the list of image paths
        if data == 'train':
            #self.paths = glob('{}/data_256/**/*.jpg'.format(data_root),
            #                  recursive=True)
            #self.mask_paths = glob('{}/mask/*.png'.format(data_root))
            self.paths = glob('{}/training/image_0/slide/**/*.png'.format(data_root), recursive=True)
            self.mask_paths = glob('{}/training/result_0/slide/**/*.png'.format(data_root))
        else:
            self.paths = glob('{}/testing/image_0/slide/000000_10/*.png'.format(data_root, data))
            self.mask_paths = glob('{}/testing/result_0/slide/000000_10/*.png'.format(data_root))

        self.N_mask = len(self.mask_paths)

    def __len__(self):
        return len(self.paths)

    def get_matching_mask(self, image_path):
        # 将image路径中的'image_0'替换为'result_0'
        mask_path = image_path.replace('image_0', 'result_0')
        if mask_path in self.mask_paths:
            return mask_path
        else:
            return None

    def __getitem__(self, index):
        img = self._load_img(self.paths[index])
        img = self.img_transform(img.convert('RGB'))
        mask_path = self.get_matching_mask(self.paths[index])
        mask = Image.open(mask_path)
        mask = self.mask_transform(mask.convert('RGB'))
        return img * mask, mask, img

    def _load_img(self, path):
        """
        For dealing with the error of loading image which is occured by the loaded image has no data.
        """
        try:
            img = Image.open(path)
        except:
            extension = path.split('.')[-1]
            for i in range(10):
                new_path = path.split('.')[0][:-1] + str(i) + '.' + extension
                try:
                    img = Image.open(new_path)
                    break
                except:
                    continue
        return img
