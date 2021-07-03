import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, Response, jsonify, request
from flask import Request
import os
import json
import requests
import base64
import io
app = Flask(__name__)


class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.eff = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)
    def forward(self, x):
        x = self.eff(x)
        x = F.softmax(x, dim=1)
        return x

def stringToGrey(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata)).convert('L')
    return image

def stringToGrey2(img_path):
    image = Image.open(open(img_path, "rb")).convert('L')
    return image

def download_image_from_url(url):
    response = requests.get(url)
    file = open("img_tmp.jpg", "wb")
    file.write(response.content)
    

@app.route('/image_rotation', methods=['POST'])
def image_rotation():
    try:
        imgurl = request.form['image']
        try:
            model_type = request.form['model_type']
        except:
            model_type = "front"

        download_image_from_url(imgurl)
        image = stringToGrey2("img_tmp.jpg")
        if transform is not None:
            image = transform(image)        
        image = image.repeat(3, 1, 1)
        image = image.unsqueeze(0).to(device)

        if model_type == "front":
            pred =  model_front(image).cpu().detach()
        else:
            pred =  model_back(image).cpu().detach()
            print('model back')

        y_pred = torch.argmax(pred).item()
        prob = pred[0, :][y_pred].item()
        os.remove("img_tmp.jpg")
        return jsonify({'class': y_pred*90, 'confidence': prob})
    except Exception as e:
        return jsonify({'class': -1, 'confidence': -1, 'error': str(e)})
    

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_front = EffNet()
    model_front.to(device)
    model_front.load_state_dict(torch.load('rotate-eff-model-front.pt'))
    model_front.eval()
    
    model_back = EffNet()
    model_back.to(device)
    model_back.load_state_dict(torch.load('rotate-eff-model-back.pt')) #back model
    model_back.eval()

    app.run("0.0.0.0", port=5022, debug=True)
