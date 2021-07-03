from utils import *
from datasets import *
from models import *
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils import TensorboardAggregator
from rich.progress import track
import random
import os
import numpy as np
from infer import process_meta_real_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_FOLDER_DATA = '/home/ubuntu/ims/rotate_classification/data/'

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def inference_prob(model, image_path, transform):
    image = Image.open(image_path).convert('L')
    if transform is not None:
        image = transform(image)
    image = image.repeat(3, 1, 1)
    image = image.unsqueeze(0).to(device)
    pred =  model(image).cpu().detach()
    y_pred = torch.argmax(pred).item()
    return y_pred, pred[0, :]


def main():
    seed_everything(2323)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    real_test_df = pd.read_csv(ROOT_FOLDER_DATA + '/real_test.csv')
    real_test_df = process_meta_real_test(real_test_df)
    
    
    model1 = EffNet()
    model1.to(device)
    model1.eval()
    model1.load_state_dict(torch.load("model_gray_eff_double_enrich.pt"))

    model2 = ResNet50()
    model2.to(device)
    model2.eval()
    model2.load_state_dict(torch.load("model_gray_resnet50_double_enrich.pt"))

    model3 = MobileNet()
    model3.to(device)
    model3.eval()
    model3.load_state_dict(torch.load("model_gray_mobile_double_enrich.pt"))

    with torch.no_grad():
        lst_pred = []
        lst_prob = []
        for (label, path) in tqdm((zip(real_test_df['labels'].values, real_test_df['paths'].values)), total=len(real_test_df)):
            y_pred_1, y_prob_1 = inference_prob(model1, path, transform)
            y_pred_2, y_prob_2 = inference_prob(model2, path, transform)
            y_pred_3, y_prob_3 = inference_prob(model3, path, transform)      
            y_prob = torch.cat((y_prob_1, y_prob_2, y_prob_3)).reshape(3,4)
            # values, indices = torch.max(y_prob, 0)
            #lst_pred.append(y_prob.argmax().item()%4)
            #lst_prob.append(y_prob.max().item())
            y_prob = (y_prob_1 + y_prob_2 + y_prob_3) / 3.0
            lst_pred.append(torch.argmax(y_prob).item())
            lst_prob.append(max(y_prob).item())
        print("Final accuracy: ", accuracy_score(lst_pred, real_test_df['labels'].values))

    real_test_df["class_preds"] = lst_pred
    real_test_df["probs"] = lst_prob
    real_test_df.to_csv('pred_test_ensemble_way2.csv', index=False)
    print(real_test_df)
if __name__ == "__main__":
    main()
