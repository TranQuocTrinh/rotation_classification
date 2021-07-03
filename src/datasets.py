import pandas as pd
import numpy as np
import os
import math
from PIL import Image, ImageOps  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class Image_Dataset(Dataset):
    def __init__(self, img_data, transform=None):
        self.transform = transform
        self.img_data = img_data
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img_name = self.img_data.loc[index, 'paths'] #os.path.join(self.folder_data, self.img_data.loc[index, 'images'])
        image = Image.open(img_name).convert('L')#.convert('RGB')
        label = self.img_data.loc[index, 'labels']
        label = torch.tensor(label)
        if self.transform is not None:
            image = self.transform(image)
        image = image.repeat(3, 1, 1)

        #paddle
        result = paddle_ocr(img_name)
        director = find_angle_bounding_box(result)
        director = torch.FloatTensor(director)
        return image, director, label



from paddleocr import PaddleOCR, draw_ocr
PADDLE = PaddleOCR(use_angle_cls=True, lang='en')
def paddle_ocr(img_path):
    result = PADDLE.ocr(img_path, cls=True)
    texts = [x[1][0] for x in result]
    boxes = [x[0] for x in result]        # (x1, y1), (x2, y2), (x3, y3), (x4, y4)

    # draw result
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    if len(result) != 0:
        scores = [line[1][1] for line in result]
    else:
        scores = []
    # im_show = draw_ocr(image, boxes, txts, scores)
    # im_show.save(name_result)
    return texts, boxes, scores

def find_angle_bounding_box(result):
    texts, boxes, scores = result
    if len(texts) == 0:
        return [0, 0, 0]

    idx_max = 0
    for i, s in enumerate(scores):
        if s > scores[idx_max]:
            idx_max = i
    box = boxes[idx_max]
    A, B, C, D = box
    
    def distance(A,B):
        x1, y1 = A
        x2, y2 = B
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist
    
    def findDirector(A, B):
        x1, y1 = A
        x2, y2 = B
        if x2-x1 == 0:
            return [0, 0, 1]
        slope = (y2-y1)/(x2-x1)
        cornet = math.atan(slope)/math.pi
        if -1/8. < cornet < 1/8.:   # horizontal-> maybe 0 or 180
            return [1, 0, 0]    
        elif -3/8. <= cornet <= -1/8. or 1/8. <= cornet <= 3/8.:    #diagonal
            return [0, 1, 0]
        elif -1/2. < cornet < -3/8. or 3/8. < cornet < 1/2.:    #vertical  90 or 270
            return [0, 0, 1]
    if distance(A, B) > distance(A, D):
        return findDirector(A, B)
    else:
        return findDirector(A, D)
  
def main():
    pass
    
if __name__ == "__main__":
    main()  
