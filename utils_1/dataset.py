from torch.utils.data import Dataset
from fake_chs_lp.random_plate import Draw
from torch import nn
import os
from torchvision.transforms import transforms
from einops import rearrange
import random
import cv2
from utils_1 import enhance, make_label
import numpy
import torch
import ocr_config
import detect_config
import re


class OcrDataSet(Dataset):
    """
    OcrDataSet is a custom Dataset class for training an OCR model to recognize license plates.
    It generates synthetic license plate images with corresponding labels, and applies data augmentation.
    """

    def __init__(self):
        super(OcrDataSet, self).__init__()
        self.dataset = []
        self.draw = Draw()
        for i in range(100000):
            self.dataset.append(1)  # Add synthetic data
        self.smudge = enhance.Smudge()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        plate, label = self.draw()
        target = []
        for i in label:
            target.append(ocr_config.class_name.index(i))
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)

        # Data augmentation (optional)
        # plate = self.data_to_enhance(plate)

        cv2.imshow("a", plate)
        cv2.imwrite("1.jpg", plate)
        cv2.waitKey()

        image = torch.from_numpy(plate).permute(2, 0, 1) / 255
        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()
        _target = torch.full(size=(15,), fill_value=0, dtype=torch.long)
        _target[: len(target)] = target

        return image, _target, target_length

    def data_to_enhance(self, plate):
        """Random smudge, Gaussian blur, Gaussian noise, and data augmentation."""
        plate = self.smudge(plate)
        plate = enhance.gauss_blur(plate)
        plate = enhance.gauss_noise(plate)
        plate, pts = enhance.augment_sample(plate)
        plate = enhance.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate


class DetectDataset(Dataset):
    """
    DetectDataset is a custom Dataset class for training a license plate detection model.
    It loads images, applies random augmentations, and returns the transformed images along with bounding box labels.
    """

    def __init__(self):
        super(DetectDataset, self).__init__()
        self.dataset = []
        self.draw = Draw()
        self.smudge = enhance.Smudge()
        root = detect_config.image_root
        root = "E:/CCPD2019/ccpd_fn"
        for image_name in os.listdir("E:/CCPD2019/ccpd_fn"):
            box = self.get_box(image_name)
            x3, y3, x4, y4, x1, y1, x2, y2 = box
            box = [x1, y1, x2, y2, x4, y4, x3, y3]
            self.dataset.append((f"{root}/{image_name}", box))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, points = self.dataset[item]
        image = cv2.imread(image_path)

        # Replace with synthetic license plate (randomly)
        if random.random() < 0.5:
            plate, _ = self.draw()
            plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
            plate = self.smudge(plate)
            image = enhance.apply_plate(image, points, plate)

        # Augment image and bounding box
        [x1, y1, x2, y2, x4, y4, x3, y3] = points
        points = [x1, x2, x3, x4, y1, y2, y3, y4]
        image, pts = enhance.augment_detect(image, points, 208)

        cv2.imshow("a", image)
        cv2.imwrite("1.jpg", image)
        cv2.waitKey()

        image_tensor = torch.from_numpy(image) / 255
        image_tensor = rearrange(image_tensor, "h w c -> c h w")
        label = make_label.object_label(pts, 208, 16)
        label = torch.from_numpy(label).float()
        return image_tensor, label

    def up_background(self, image):
        """Apply Gaussian blur, noise, and random cropping to the background."""
        image = enhance.gauss_blur(image)
        image = enhance.gauss_noise(image)
        image = enhance.random_cut(image, (208, 208))
        return image

    def data_to_enhance(self, plate):
        """Random smudge, Gaussian blur, Gaussian noise, and data augmentation."""
        plate = self.smudge(plate)
        plate = enhance.gauss_blur(plate)
        plate = enhance.gauss_noise(plate)
        plate, pts = enhance.augment_sample(plate)
        plate = enhance.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate

    def get_box(self, name):
        """Extract bounding box coordinates from the image name."""
        name = re.split("[.&_-]", name)[7:15]
        name = [int(i) for i in name]
        return name


if __name__ == "__main__":
    data_set = OcrDataSet()
    for i in range(1000):
        data_set[1]
