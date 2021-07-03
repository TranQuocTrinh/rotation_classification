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
from torch import optim
import random
import os
import numpy as np
from infer import eval_real_test
from sklearn.model_selection import train_test_split

ROOT_FOLDER_DATA = '../'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, criterion, optimizer):
    logs_file = f"./tb_logs/final_raw/"
    writer = SummaryWriter(logs_file)
    agg = TensorboardAggregator(writer)
    model.train()
    running_loss = .0
    total = 0
    correct = 0
    bar = track(train_loader, total=len(train_loader), description='Training....')
    bar = tqdm(train_loader, total=len(train_loader))

    for data_, director_, target_ in bar: 
        data_, director_, target_ = data_.to(device), director_.to(device), target_.to(device)# on GPU
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(data_, director_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == target_).item()
        total += target_.size(0)

    accuracy = 100 * correct / total
    loss_score = running_loss/len(train_loader)
    return accuracy, loss_score


def eval(model, val_loader, criterion, optimizer):
    model.eval()
    batch_loss = 0
    total_t = 0
    correct_t = 0
    bar = track(val_loader, total=len(val_loader), description='Valid....')
    for data_t,director_t, target_t in bar:
        data_t, director_t, target_t = data_t.to(device), director_t.to(device), target_t.to(device)# on GPU
        outputs_t = model(data_t, director_t)
        loss_t = criterion(outputs_t, target_t)
        batch_loss += loss_t.item()
        _,pred_t = torch.max(outputs_t, dim=1)
        correct_t += torch.sum(pred_t==target_t).item()
        total_t += target_t.size(0)
    accuracy = 100 * correct_t / total_t
    loss_score = batch_loss/len(val_loader)
    return accuracy, loss_score

def main():
    seed_everything(2323)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 16
    """
    select_columns = ['labels', 'paths']
    
    # flickr8k_df = pd.read_csv(ROOT_FOLDER_DATA + 'metadata_flickr8k.csv').sample(frac=1) # flickr8k
    flickr30k_df = pd.read_csv(ROOT_FOLDER_DATA + 'metadata_flickr30k.csv')#.sample(frac=0.1)  # flickr30k
    label_df = pd.read_csv(ROOT_FOLDER_DATA + 'metadata_hand_label.csv')#.sample(frac=0.1)  # hand_label
    enrich_df = pd.read_csv(ROOT_FOLDER_DATA + 'metadata_enrich.csv')#.sample(frac=0.1)   # enrich
    enrich_df2 = pd.read_csv(ROOT_FOLDER_DATA + 'metadata_enrich2.csv')#.sample(frac=0.1)     # enrich2
    wrong_df = pd.read_csv(ROOT_FOLDER_DATA + 'metadata_test_wrong.csv')#.sample(frac=0.1)  # test wrong

    data = pd.concat([
                    flickr30k_df[select_columns], 
                    label_df[select_columns], 
                    enrich_df[select_columns], 
                    enrich_df2[select_columns],
                    wrong_df[select_columns]]).reset_index(drop=True)
    """
    data = pd.read_csv(ROOT_FOLDER_DATA + "metadata_sample.csv")
    train_df, val_df = train_test_split(data, test_size=0.01, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print('Train:', train_df.shape)
    print('Validation:', val_df.shape)

    train_dataset = Image_Dataset(img_data=train_df, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = Image_Dataset(img_data=val_df, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model = EffNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()#.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    n_epochs = 5
    real_test_acc = -np.Inf

    # real test
    real_test_df = pd.read_csv(ROOT_FOLDER_DATA + 'metadata_test.csv')

    for epoch in range(1, n_epochs+1):
        print('\nEpoch = [{}]/[{}]\n'.format(epoch, n_epochs))
        t_acc, t_loss = train(model, train_loader, criterion, optimizer)
        print(f'\ntrain loss: {t_loss:.4f}, train acc: {t_acc:.4f}')
        with torch.no_grad():
            v_acc, v_loss = eval(model, val_loader, criterion, optimizer)
            print(f'validation loss: {v_loss:.4f}, validation acc: {v_acc:.4f}\n')

        with torch.no_grad():
            test_acc, lst_false, pred, prob = eval_real_test(model, real_test_df, transform)
            print(f'REAL TEST acc: {test_acc:.4f}\n')
            network_learned = test_acc > real_test_acc
            # Saving the best weight
            if network_learned:
                real_test_acc = test_acc
                torch.save(model.state_dict(), 'model_gray_eff_cat_paddle.pt')
                print('Detected network improvement, saving current model')
            
    print('Best accuracy is {}'.format(real_test_acc))

if __name__ == "__main__":
    main()
