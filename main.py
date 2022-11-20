import clip
import torch
from nn_model import *
from PIL import Image

def dummy_train(device):
    model = Network(device=device)
    if device == 'cuda':
        model.cuda()
    
    epochs = 1
    optimizer = torch.optim.Adam(lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'