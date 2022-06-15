from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

class MyImageDataGenerator:
    def __init__(self):
        self.datagen = ImageDataGenerator(horizontal_flip=True, preprocessing_function=preprocess_input)

    def flow(self, x, y=None, batch_size=32, shuffle=False, sample_weight=None,
             seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = self.datagen.flow(x, y, batch_size, shuffle, sample_weight,
                               seed, save_to_dir, save_prefix, save_format, subset)

    def flow_from_directory(self, directory, target_size=(112, 112), color_mode='rgb', classes=None,
                          class_mode='categorical', batch_size=32, shuffle=False, seed=None,
                          save_to_dir=None, save_prefix='', save_format='png', follow_links=False,
                          subset=None, interpolation='nearest'):
        batches = self.datagen.flow_from_directory(directory, target_size, color_mode, classes,
                          class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format,
                          follow_links, subset, interpolation)

        while True:
            x_batch, y_batch = next(batches)

            yield [x_batch, y_batch], y_batch

