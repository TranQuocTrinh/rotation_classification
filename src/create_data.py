import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd


def rotate_image(img, angle):   # same client
    if angle == 0:
        img_rotate = img.copy()
    elif angle == 90:
        img_rotate = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        img_rotate = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        img_rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img_rotate


def create_data(source_folder):
    filenames = [x for x in os.listdir(source_folder) if int(x.split('.')[0].split('_')[0]) not in {90, 180, 270}]
    # random index image

    #idxs = [random.randint(0, len(filenames)) for i in range(2697)]  # 8091/3
    idxs = list(np.random.choice(len(filenames), size=len(filenames), replace=False))

    bar = tqdm(idxs)
    for i in bar:
        filename = filenames[i]
        img = cv2.imread(source_folder + filename)
        
        img90 = rotate_image(img, 90)
        img180 = rotate_image(img, 180)
        img270 = rotate_image(img, 270)

        cv2.imwrite(source_folder + '90_' + filename, img90)
        cv2.imwrite(source_folder + '180_' + filename, img180)
        cv2.imwrite(source_folder + '270_' + filename, img270)
        bar.set_description("index: {}".format(i))
    

source_folder = '/home/ubuntu/ims/rotate_classification/data/flickr30k/'
create_data(source_folder=source_folder)
images_path = [source_folder + x for x in os.listdir(source_folder)]

images = [x.split('/')[-1] for x in images_path]
labels = []
for name in images:
    if name.split('_')[0] == "90":
        labels.append('1')
    elif name.split('_')[0] == "180":
        labels.append('2')
    elif name.split('_')[0] == "270":
        labels.append('3')
    else:
        labels.append('0')

df = pd.DataFrame({'images': images, 'labels': labels, 'paths': images_path})
df.to_csv('/home/ubuntu/ims/rotate_classification/data/metadata_'+ source_folder.split('/')[-2] + '.csv', index=False)


# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.2)
# train.reset_index().to_csv('data/train.csv', index=False)
# test.reset_index().to_csv('data/test.csv', index=False)
