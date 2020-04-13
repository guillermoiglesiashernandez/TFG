import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data
from skimage.color import rgb2gray

image = Image.open("C:/Users/Guillermo/Desktop/Imagenes/danbooru-images/danbooru-images/0051/2201051.jpg")
print(np.shape(image))
image = rgb2gray(image)
plt.imshow(image)
plt.show()

print(np.shape(image))
