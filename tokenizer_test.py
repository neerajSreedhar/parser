import clip
from clip.simple_tokenizer import SimpleTokenizer
import torch
import nn_model
import pandas as pd
from PIL import Image
import numpy as np

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    image = preprocess(Image.open("dog.jpeg")).unsqueeze(0).to(device)
    text2tokens = {}
    strings = ['push watermelon and rotate it',
                'place both camera and banana above orange',
                'push apple and clock forward',
                'put orange, watermelon and strawberry in a line',
                'push all food forward']
    text = clip.tokenize(strings).to(device)
    print(text)
    text = text.cpu().numpy()
    _tokenizer = SimpleTokenizer()
    print(_tokenizer.decode(text[0]))
    text2tokens['strings'] = strings
    tokens = []
    for i, txt in enumerate(strings):
        tokens.append(np.array2string(text[i]))
    text2tokens['tokens'] = tokens
    print(tokens)
    tok_res = pd.DataFrame(text2tokens)
    tok_res.to_csv('tokenization_results.csv')
    '''with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(f'image features shape: {image_features.shape}')
    print(f'text features shape: {text_features.shape}')
    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]'''
