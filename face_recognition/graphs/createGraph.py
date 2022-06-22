import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from loadContentFiles import *

"""
    Create the Roc curve graph
"""

#Load config file
cfgData = load_yaml('../config.yml')
data = np.load(cfgData['evaluation-data'])
data2 = np.load('E:\\Projeto-FaceRecognition\\face_recognition\\evaluationsResult\\ICBRW\\Baseline\\22062022_2120\\Baseline.npz')
TPR = data['x']
FPR = data['y']

TPR2 = data2['x']
FPR2 = data2['y']

print(data['w'])
print(data['z'])

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
plt.plot(FPR2, TPR2, linewidth=2)
plt.plot(FPR, TPR, linewidth=2)

# customization
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(title='AUC', title_fontsize=13, labels=[str(round(float(data2['z']), 3)) + ' Baseline', str(round(float(data['z']), 3)) + ' Align'], loc='lower right')
#plt.legend(title='AUC', title_fontsize=13, labels=[str(round(float(data['z']), 3)) + ' Align-L2-CosineSim'], loc='lower right')
plt.savefig('RocCurve/ICBRW/BaselineAlign2.png')
#plt.show()

