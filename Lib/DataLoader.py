import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class HW1DataSet(Dataset):

    def __init__(self, path_to_folders, images_txt_file, load_images=False):
        with open(images_txt_file, 'r') as f:
            self._file_names = f.readlines()
        self.load_images = load_images
        self._root = path_to_folders
        self._image_path = self._root + "/images/"
        self._label_path = self._root + "/bboxes_labels/"
        self._to_tensor = transforms.ToTensor()
        self.labels = []
        for image in self._file_names:
            file_name = image.split(',')[0]
            with open(self._label_path + file_name + '.txt', 'r') as f:
                labels = [x.split() for x in f.readlines()]
            self.labels.append(labels)
        if self.load_images:
            self._images = [self._to_tensor(Image.open(self._image_path + image)) for image in self._file_names]
        else:
            self._images = []

    def __getitem__(self, item):
        if self.load_images:
            sample = self._images[item]
        else:
            image = self._file_names[item]
            sample = self._to_tensor(Image.open(self._image_path + image))
        label, x_center, y_center, w, h = self.labels[item]
        return sample, (torch.tensor(label), torch.tensor([x_center, y_center, w, h]))

    def __len__(self):
        return len(self._file_names)