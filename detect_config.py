from models_1.detect_net import DetectNet
import torch

image_root = "E:/CCPD2019/ccpd_base"
batch_size = 128
weight = "weights/detect_net.pt"
epoch = 500
net = DetectNet
device = "cuda:0"
confidence_threshold = 0.9


device = torch.device(device if torch.cuda.is_available() else "cpu")
