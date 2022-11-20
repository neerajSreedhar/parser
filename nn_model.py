import clip
import torch
from PIL import Image

class Network(torch.nn.Module):
    def __init__(self, clip_model='ViT-B/32', device='cpu', n_hidden=196, n_layers=4, drop_prob=0.5):
        super().__init__()
        self.device = device
        model, preprocess = clip.load(clip_model, device=self.device)
        self.clipModel = model
        self.preprocess = preprocess
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lstm = torch.nn.LSTM(1024, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
    
    def forward(self, x, hidden):
        '''
            Passing x as a tuple (Image, instruction).
            Input shape to LSTM: (1, 1024).
            Output shape from Softmax: (1, 196)
        '''
        image = self.preprocess(x[0]).unsqueeze(0).to(self.device)
        text = clip.tokenize([x[1]]).to(self.device)

        with torch.no_grad():
            image_features = self.clipModel.encode_image(image)
            text_features = self.clipModel.encode_text(text)

        concat = torch.cat((image_features, text_features), dim=1).float()
        lstm_out, hidden = self.lstm(concat, hidden)
        out = torch.nn.functional.softmax(lstm_out, dim=1)
        return out, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.device == 'cuda':
            w1 = weight.new(self.n_layers, self.n_hidden).zero_().cuda()
            w2 = weight.new(self.n_layers, self.n_hidden).zero_().cuda()
        else:
            w1 = weight.new(self.n_layers, self.n_hidden).zero_()
            w2 = weight.new(self.n_layers, self.n_hidden).zero_()       
        hidden = (w1, w2)
        return hidden

def test_clip(device='cpu'):
    model, preprocess = clip.load('ViT-B/32', device=device)
    image = preprocess(Image.open("dog.jpeg")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(f'image features shape: {image_features.shape}')
    print(f'text features shape: {text_features.shape}')
    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

if __name__ == '__main__':
    #print(device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #test_clip(device=device)
    img = Image.open('dog.jpeg')
    instruction = 'it is a dog'
    model = Network(device=device)
    if device == 'cuda':
        model.cuda()
    hidden_ini = model.init_hidden()
    out, hidden = model((img, instruction), hidden_ini)
    print(out.shape)