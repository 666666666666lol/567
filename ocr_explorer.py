from models_1.ocr_net import OcrNet
import ocr_config as config
import torch
import cv2
import numpy as np
import os


class Explorer:
    """
    The Explorer class is responsible for loading a pre-trained OCR model,
    processing input images, and performing optical character recognition (OCR)
    to extract text from images of license plates. It also handles removing
    duplicate characters in the recognized text.
    """

    def __init__(self, is_cuda=False):
        self.device = config.device
        self.net = OcrNet(config.num_class)
        if os.path.exists(config.weight):
            self.net.load_state_dict(torch.load(config.weight, map_location="cpu"))
            print("Successfully loaded model parameters")
        else:
            raise RuntimeError("Model parameters are not loaded")
        self.net = self.net.to(self.device).eval()

    def __call__(self, image):
        with torch.no_grad():
            image = torch.from_numpy(image).permute(2, 0, 1) / 255
            image = image.unsqueeze(0).to(self.device)
            out = self.net(image).reshape(-1, 70)
            out = torch.argmax(out, dim=1)
            out = out.cpu().numpy().tolist()
            c = ""
            for i in out:
                c += config.class_name[i]
            return self.deduplication(c)

    def deduplication(self, c):
        """Remove duplicate characters"""
        temp = ""
        new = ""
        for i in c:
            if i == temp:
                continue
            else:
                if i == "*":
                    temp = i
                    continue
                new += i
                temp = i
        return new
