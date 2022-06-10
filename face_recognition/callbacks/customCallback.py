import keras.callbacks
import matplotlib.pyplot as plt
from callbacks.lfw_eval import lfw_eval_callback

class LossAndAccuracySaveImage(keras.callbacks.Callback):
    def __init__(self, **kargs):
        super(LossAndAccuracySaveImage,self).__init__(**kargs)
        self.epoch_accuracy = []
        self.epoch_loss=[]
        self.epoch_valAccuracy=[]
        self.epoch_valloss=[]

    def on_epoch_begin(self, epoch, logs={}):
        # Things done on beginning of epoch.
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_accuracy.append(logs.get("accuracy"))
        self.epoch_loss.append(logs.get("loss"))
        self.epoch_valloss.append(logs.get("val_loss"))
        self.epoch_valAccuracy.append(logs.get("val_accuracy"))
        if(logs.get("val_loss") < self.epoch_valloss[len(self.epoch_valloss)-2]):

            # loss
            plt.plot(self.epoch_loss, label='train loss')
            plt.plot(self.epoch_valloss, label='val loss')
            plt.legend()
            #plt.show()
            plt.savefig('graphs/LossVal_loss_e'+str(epoch))
            plt.clf()
            # accuracies
            plt.plot(self.epoch_accuracy, label='train acc')
            plt.plot(self.epoch_valAccuracy, label='val acc')
            plt.legend()
            #plt.show()
            name='AccVal_acc_e'+str(epoch)
            print(name)
            plt.savefig('graphs/'+ name)
            plt.clf()


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