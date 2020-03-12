import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

tagList = []
tagCont = []
x = []
y = []

with open("Tags/Tags.txt", 'r', encoding="utf8") as f:
    tagList = json.load(f)
with open("Tags/Values.txt", 'r', encoding="utf8") as f:
    tagCont = json.load(f)
salida = open("Tags/TagValues.txt", 'a', encoding="utf8")

for element in tagCont:
    if element >= 40000:
        x.append(tagList[tagCont.index(element)]['name'])
        y.append(element)
        salida.write(tagList[tagCont.index(element)]['name'] + "\t" + str(element) + "\n")

y_pos = np.arange(len(x))

plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('Nº Imgs')
plt.title('Label title')

#plt.show()