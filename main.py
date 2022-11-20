import clip
import torch
from nn_model import *
from PIL import Image

def dummy_train(device, inputs, targets):
    model = Network(device=device, n_hidden=77)
    if device == 'cuda':
        model.cuda()
    model.train()
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        h = model.init_hidden()
        for x, y in zip(inputs, targets):
            h = tuple(each.data for each in h)
            model.zero_grad()
            out, h = model(x, h)
            loss = criterion(out, y)
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            print(f'Loss: {loss.item()}')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.open('dog.jpeg')
    instruction = 'it is a dog'
    y = clip.tokenize([instruction]).to(device).float()
    dummy_train(device=device, inputs=[(img, instruction)], targets=[y])