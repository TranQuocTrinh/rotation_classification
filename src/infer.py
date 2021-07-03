from utils import *
from models import *
from datasets import Image_Dataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils import TensorboardAggregator
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import json
from rich.progress import track
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from datasets import paddle_ocr, find_angle_bounding_box

ROOT_FOLDER_DATA = '../'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_real_test(model, df, transform):
    print('Model evaluation on real test...')
    results = [inference(model, x , transform) for x in tqdm(df['paths'].values)]
    pred = [r[0] for r in results]
    prob = [r[1] for r in results]

    result = np.array(pred) == df["labels"].values
    acc = 100 * sum(result) / len(pred)
    # json.dump(pred, open('y_pred_model.json','w'))
    return acc, list(df.loc[result, "product_id"]), pred, prob


def inference(model, image_path, transform):
    image = Image.open(image_path).convert('L')#convert('RGB')
    if transform is not None:
        image = transform(image)
    
    image = image.repeat(3, 1, 1)
    image = image.unsqueeze(0).to(device)

    director = paddle_ocr(image_path)
    director = find_angle_bounding_box(director)
    director = torch.FloatTensor(director).unsqueeze(0).to(device)
    
    pred =  model(image, director).cpu().detach()
    y_pred = torch.argmax(pred).item()
    prob = pred[0, :][y_pred].item()
    return (y_pred, prob)

"""
def process_meta_real_test(df):
    df = df.rename(columns={"rotation": "labels", "frontimg": "images"})
    def split_path_image(url):
        return url.split('/')[-1]
    def fix_label(label):
        class_labels = {0: 0, 90: 1, 180: 2, 270: 3}
        return class_labels[label]
    
    df["images"] = df["images"].apply(split_path_image)
    df["labels"] = df["labels"].apply(fix_label)
    df['paths'] = [ROOT_FOLDER_DATA + 'real_imgs/' + x for x in df["images"]]
    return df
"""

def gama_statistic(result_df, error=0.005):
    def find_gama(acc, begin=0):
        for i in track(range(begin, 10000), description='Find gama ...'):
            gama = (i+1)/10000
            idx = [i for (i, x) in enumerate(result_df.probs.values >= gama) if x == True]
            if len(idx) != 0:
                accuracy = accuracy_score(result_df.loc[idx, 'labels'].values, result_df.loc[idx, 'class_preds'])
                if accuracy >= acc:
                    return gama, len(idx)
        
    dct = {'gama':[], 'acc':[], 'sample': [], 'total_sample':[]}
    
    begin = 0
    accuracy = [round(0.9 + 0.005*i, 4) for i in range(21)]  #0.900 -> 1 with jump 0.005
    for acc in accuracy:
        try:
            gama, sample = find_gama(acc, begin)
            begin = int(gama*10000)
        except:
            gama, sample = 'none', 'none'
            begin = 10000
        dct['acc'].append(acc)
        dct['gama'].append(gama)
        dct['sample'].append(sample)
        dct['total_sample'].append(len(result_df))
    return pd.DataFrame(dct)



def main():
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = EffNet()
    model.to(device)
    model.load_state_dict(torch.load('model_gray_eff_double_enrich.pt'))
    model.eval()

    df = pd.read_csv(ROOT_FOLDER_DATA + 'real_test.csv')
    df = process_meta_real_test(df)
    acc, id_false, y_pred, prob = eval_real_test(model, df, transform)
    df['class_preds'] = y_pred
    df['probs'] = prob
    df.to_csv('pred_test.csv', index=False)
    print('Real test accuracy: {}'.format(acc))
    """

    # confusion matrix model ensemble
    df = pd.read_csv('pred_test_ensemble.csv')
    print('Real test accuracy:', accuracy_score(df.labels.values, df.class_preds.values))
    conf_matrix = confusion_matrix(df.labels.values, df.class_preds.values, labels=[0, 1, 2, 3])

    dct = {'': [0, 90, 180, 270]}
    for i in range(4):
        dct[str(i*90)] = conf_matrix[:, i]
    
    conf_matrix = pd.DataFrame(dct)
    print(conf_matrix)
    conf_matrix.to_csv('confusion_matrix_ensemble.csv', index=False)



    # # Statistical gama
    # df = pd.read_csv('pred_test.csv')
    # print('Statistical... ')
    # statistical = gama_statistic(df)
    # statistical.to_csv('gama_statistic.csv', index=False)
    # print(statistical)


if __name__ == "__main__":
    main()
