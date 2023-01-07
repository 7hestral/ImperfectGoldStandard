import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms

import os

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter

class myDataSet(Dataset):
    def __init__(self, biased_csv_folder, all_image_path, loader=default_loader, transform=None):
        normal_files = pd.read_csv(os.path.join(biased_csv_folder, 'normal_names.csv'))['name'].to_list()
        pneu_files = pd.read_csv(os.path.join(biased_csv_folder, 'pneu_names.csv'))['name'].to_list()

        normal_items = [(os.path.join(all_image_path, item), 0) for item in normal_files]
        pneu_items = [(os.path.join(all_image_path, item), 1) for item in pneu_files]

        self.samples = normal_items + pneu_items
        self.loader = loader
        self.transform = transform
        self.targets = [x[1] for x in self.samples]
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # print(path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    def __len__(self):
        return len(self.samples)


class ContrastBrightness(object):
    """Image pre-processing.

    alpha = 1.0 # Simple contrast control [1.0-3.0]
    beta = 0    # Simple brightness control [0-100]
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image, ):
        image = np.array(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                image[y, x] = np.clip(self.alpha * image[y, x] + self.beta, 0, 255)

                return Image.fromarray(np.uint8(image) * 255)
class HistEqualization(object):
    """Image pre-processing.

    Equalize the image historgram
    """

    def __call__(self, image):
        return ImageOps.equalize(image, mask=None)
class SmothImage(object):
    """Image pre-processing.

    Smooth the image
    """

    def __call__(self, image):
        return image.filter(ImageFilter.SMOOTH_MORE)
if __name__ == '__main__':
    transform = transforms.Compose([

        transforms.Resize([224, 224]),
        # ContrastBrightness(1.2, 25),
        HistEqualization(),
        # SmothImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2365, 0.2365, 0.2365))
    ])
    ds = myDataSet('./0_10_trial3_csv/train', './all_images/train', transform=transform)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    i=0
    for img, label in dataloader:
        if i == 549:
            plt.imshow(img[0].permute(1,2,0))
            plt.show()
            print(label[0])
            break
        i+=1
