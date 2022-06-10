import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from loadContentFiles import *

cfgData = load_yaml('../config.yml')
data = np.load(cfgData['evaluation-data'])
TPR = data['x']
FPR = data['y']

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
plt.plot(FPR, TPR, linewidth=2)
# customization
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.legend(title='Auc', title_fontsize=13, labels=[round(float(data['z']), 2)])
#plt.savefig('Align-L2-CousineSim.png')
plt.show()