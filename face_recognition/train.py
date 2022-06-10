import os
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import (
    BatchNormalization
)
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from callbacks.customCallback import LossAndAccuracySaveImage, LFWEvaluation
from loadContentFiles import load_yaml
from model import ResNet50WithSoftmax
import tensorflow as tf
from datetime import datetime

#Load config file
cfgData = load_yaml('config.yml')

# re-size all the images to this
IMAGE_SIZE = [cfgData['inputSize'], cfgData['inputSize']]
CNN_ARCH = cfgData['backbone']
EXPERIMENT_NAME = cfgData['experiment']

train_path = cfgData['train-path']
valid_path = cfgData['validation-path']


folders = glob(train_path + "/*")
model=ResNet50WithSoftmax(cfgData, len(folders))

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=cfgData['learning-rate']),
    metrics=['accuracy']
)


train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(cfgData['inputSize'], cfgData['inputSize']),
                                                 batch_size=cfgData['batch-size'],
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size=(cfgData['inputSize'], cfgData['inputSize']),
                                            batch_size=cfgData['batch-size'],
                                            class_mode='categorical')

checkpoint_timestamp = datetime.now().strftime("%d%m%Y_%H%M")
path = os.path.join('checkpoints', CNN_ARCH, EXPERIMENT_NAME, checkpoint_timestamp)
if not os.path.exists(path):
    os.makedirs(path)
modelCheckpoint_callback = ModelCheckpoint(
    path,
    save_weights_only=True,
    save_best_only=True,
    verbose=1
    )

eval_lfw_verif_mode = LFWEvaluation()

# fit the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=cfgData['epoch'],
    callbacks=[modelCheckpoint_callback, eval_lfw_verif_mode],
)

model.save('resnet50_softmax_casia.h5')
np.save('checkpoints/ResNet50/Baseline/history1.npy', r.history)
np.save(path +'/history1.npy', r.history)

