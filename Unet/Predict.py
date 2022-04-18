import torch
from torch import nn, optim
from torchvision import transforms
import ISBI_dataloader as dl
import Unet_Structure


def predict(model, image):
    out = model(image)
    out = transforms.Resize((572, 572))(out)
    return out
