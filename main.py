import clip
import torch
import nn_model
from PIL import Image

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'