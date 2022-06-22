import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from loadContentFiles import *

"""
    Create Loss curve or accuracy curve graph
"""
cfgData = load_yaml('../config.yml')
data = np.load(cfgData['model-data'], allow_pickle=True).item()
url2="E:\\Projeto-FaceRecognition\\face_recognition\\checkpoints\\Baseline\\history.npy"
data2=np.load(url2, allow_pickle=True).item()

sns.set_style('darkgrid')  # darkgrid, white grid, dark, white and ticks
sns.color_palette('pastel')
plt.rc('axes', titlesize=18)  # fontsize of the axes title
plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
plt.rc('legend', fontsize=13)  # legend fontsize
plt.rc('font', size=13)  # controls default text sizes
plt.figure(figsize=(7, 7), tight_layout=True)

# plotting
ab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#plt.plot(range(len(ab)), data['loss'])
#plt.plot(range(len(ab)), data['val_loss'])
#plt.plot(range(len(ab)), data2['loss'])
#plt.plot(range(len(ab)), data2['val_loss'])
plt.plot(range(len(ab)), data['accuracy'])
plt.plot(range(len(ab)), data['val_accuracy'])
plt.plot(range(len(ab)), data2['accuracy'])
plt.plot(range(len(ab)), data2['val_accuracy'])
plt.xticks(range(len(ab)), ab)
# customization
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
#plt.ylabel('Loss')
#plt.title('Loss Curves')
plt.legend(title='Curves', title_fontsize=13, labels=['Train - Align','Validation - Align', 'Train - Baseline','Validation - Baseline'])
#plt.savefig('TrainCurve/Align/Accuracy_CurveAlign.png')
plt.savefig('TrainCurve/Accuracy_CurveAlignBaseline.png')
#plt.show()
