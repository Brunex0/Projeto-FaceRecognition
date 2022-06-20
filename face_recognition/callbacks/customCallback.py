import keras.callbacks
import matplotlib.pyplot as plt
from callbacks.lfw_eval import lfw_eval_callback

class LFWEvaluation(keras.callbacks.Callback):
    """
        Callback to evaluate on LFW at the end of each epoch
    """
    def __init__(self, patience=0):
        super(LFWEvaluation, self).__init__()
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 9

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        lfw_eval_callback(self.model)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            plt.clf()