import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

epochAccuracyArray=[]
with open('C:/Users/Guillermo/Desktop/Merge/accuracyPerformance.txt') as file1:
    for line in file1.readlines():
        epochAccuracyArray.append(float(line))

epochAccuracyArrayTest=[]
with open('C:/Users/Guillermo/Desktop/Merge/accuracyTestPerformance.txt') as file2:
    for line in file2.readlines():
        epochAccuracyArrayTest.append(float(line))

epochLossArray=[]
with open('C:/Users/Guillermo/Desktop/Merge/lossPerformance.txt') as file3:
    for line in file3.readlines():
        epochLossArray.append(float(line))

epochAccuracyArray = np.array(epochAccuracyArray)
epochAccuracyArrayTest = np.array(epochAccuracyArrayTest)
epochLossArray = np.array(epochLossArray)

plt.subplot(311)
x_plot = np.arange(epochLossArray.shape[0])
plt.tight_layout()
plt.xlabel('Loss means')
plt.plot(x_plot, epochLossArray)

plt.subplot(312)
plt.tight_layout()
plt.xlabel('Accuracy means')
plt.plot(x_plot, epochAccuracyArray)

plt.subplot(313)
plt.tight_layout()
plt.xlabel('Accuracy tests means')
plt.plot(x_plot, epochAccuracyArrayTest)

filename = 'C:/Users/Guillermo/Desktop/Merge/MergeResults.png'
plt.savefig(filename)
plt.close()