import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cont = 0
name = ""

for folder in glob.glob("C:/Users/Guillermo/Desktop/Imagenes/danbooru-images/danbooru-images/*"):
    print(folder)
    folderName = os.path.basename(folder)

    os.makedirs("BWImages/" + folderName)

    for img in glob.glob(folder + "/*.jpg"):
        name = os.path.basename(img)
        image = Image.open(img)
        if (np.asarray(image).shape == (512, 512)):
            image.save("BWImages/" + folderName + "/" + name)
        else:
            image.convert('RGB').convert('L').save("BWImages/" + folderName + "/" + name)
        cont +=1

print("Imagenes tratadas: " + str(cont))