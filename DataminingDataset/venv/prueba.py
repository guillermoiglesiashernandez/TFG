import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

i = 0
cont = 0
error = 0
wallcomic = 0

for myFile in glob.glob('MetadataImgCleaned/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        if int(element['image_width']) < 500 and int(element['image_height']) < 500:
            print (element)
