from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    GlobalAveragePooling2D,
    Layer,
)
from tensorflow.keras.layers import (
    BatchNormalization
)
from tensorflow.keras import regularizers


def ResNet50WithSoftmax(cfgData, numClasses):
    """
    Crete a CNN model with ResNet50 as backbone and softmax as activation function
    :param cfgData: the data from config file
    :param numClasses: the number of classes of the dataset
    :return: The model
    """
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
