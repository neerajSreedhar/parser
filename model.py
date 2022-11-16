import clip
import torch
from PIL import Image

class Network(torch.nn.Module):
    def __init__(self, clip_model='ViT-B/32', device='cpu'):
        super().__init__()
        self.device = device
        model, preprocess = clip.load(clip_model, device=self.device)
        self.clipModel = model
        self.preprocess = preprocess

    
    def forward(self, x):
        image = self.preprocess(x['img']).unsqueeze(0).to(self.device)
        text = clip.tokenize([x['instruction']]).to(self.device)

        with torch.no_grad():
            image_features = self.clipModel.encode_image(image)
            text_features = self.clipModel.encode_text(text)

        concat = torch.cat((image_features, text_features), dim=1)
        return concat 

def test_clip():
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
    #test_clip()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.open('dog.jpeg')
    instruction = 'it is a dog'
    model = Network(device=device)
    
    print(model({'img':img, 'instruction':instruction}).shape)