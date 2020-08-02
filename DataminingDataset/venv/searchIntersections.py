import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

enc = 0

for myFile in glob.glob('C:/Users/Guillermo/Desktop/TFG/Archivos Grandes/Metadata Transitorio/MetadataImgCleaned/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        enc=0
        for tag in element['tags']:
            if tag['name'] == "breasts":
                enc +=1
            elif tag['name'] == "blush":
                enc +=1
            if(enc==2):
                print(element['id'])
                enc+=1