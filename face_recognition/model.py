from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    GlobalAveragePooling2D,
    Layer,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    BatchNormalization
)
from tensorflow.keras import regularizers
import tensorflow as tf
from keras import backend as K



def ResNet50WithSoftmax(cfgData, numClasses):
    IMAGE_SIZE = [cfgData['inputSize'], cfgData['inputSize']]
    # ResNet50 backbone
    backbone = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # Put backbone layers trainable
    for layer in backbone.layers:
        layer.trainable = True

    backbone.summary()

    x = BatchNormalization()(backbone.output)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=cfgData['dropout-rate'])(x)

    x = Flatten()(x)

    prediction = Dense(numClasses, activation='softmax',
                       activity_regularizer=regularizers.l2(cfgData['scale-l2-regularizer']))(x)

    # create a model object
    model = Model(inputs=backbone.input, outputs=prediction)
    return model

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

def create_simple_arcface_model(cfgData, n_out):

    input_tensor = Input(shape=(cfgData['inputSize'],cfgData['inputSize'],3))
    #x = inputs = Input([size, size, channels], name='input_image')
    output_tensor = Input(shape=n_out)
    # ResNet50 backbone
    backbone = ResNet50(input_shape= input_tensor.shape[1:], weights='imagenet', include_top=False)(input_tensor)

    x = BatchNormalization()(backbone)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=cfgData['dropout-rate'])(x)
    x = Flatten()(x)
    #labels = Input([], name='label')
    output = ArcFace(n_classes=n_out, s=6.0, m=2.0)([x, output_tensor])
    #output = ArcFace(n_classes=n_out, s=6.0, m=2.0)(x,labels)
    model = Model([input_tensor, output_tensor], output)
    #model = Model((inputs, labels), logist, name=name)
    return model