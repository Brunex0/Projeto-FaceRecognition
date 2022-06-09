import os
import shutil
import random
from glob import glob
from sklearn.model_selection import train_test_split
from loadContentFiles import load_yaml

#Load config file
cfgData = load_yaml('../config.yml')


DATASET_PATH = cfgData['database-path']
DESTINATION_PATH = cfgData['destination-path']
if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH)


train_path = os.path.join(DESTINATION_PATH, 'train')
test_path = os.path.join(DESTINATION_PATH, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
print('**Starting**')
ids = glob(DATASET_PATH)
for id_path in ids:
    id_num = id_path.split('/')[-1]
    #print('__id__ ', id_num)
    id_imgs = glob(os.path.join(id_path, '*.jpg'))

    os.makedirs(os.path.join(train_path, id_num), exist_ok=True)
    os.makedirs(os.path.join(test_path, id_num), exist_ok=True)
    for img_i, img_path in enumerate(id_imgs):
        img_name = img_path.split('/')[-1]
        if random.random() > 0.8:
            shutil.copy(img_path, os.path.join(test_path, id_num, img_name))
        else:
            shutil.copy(img_path, os.path.join(train_path, id_num, img_name))
