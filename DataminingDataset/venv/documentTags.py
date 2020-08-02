import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

i = 0
cont = 0

for myFile in glob.glob('C:/Users/Guillermo/Desktop/TFG/Archivos Grandes/Metadata Transitorio/MetadataStructured/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        if(element['score']=="0"):
            cont += 1
        i+=1

print("\n\n\n" + str(cont))
print(str(i))