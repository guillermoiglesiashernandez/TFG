import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

i = 0
cont = 0
series = 0
collections = 0
enc = False

for myFile in glob.glob('C:/Users/Guillermo/Desktop/TFG/Archivos Grandes/Metadata Transitorio/MetadataImgCleaned/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        if element['pools'] != []:
            for tag in element['pools']:
                if tag == 'series':
                    enc = True
                    series+=1
                if tag == 'collection':
                    enc = True
                    collections+=1
            #if enc == False:
            print(element['id'])
            print(element['pools'])
            folder = '%04d' % (int(element["id"]) % 1000)

            img = mpimg.imread("C:/Users/Guillermo/Desktop/TFG/Archivos Grandes/Dataset/danbooru-images/danbooru-images/" + folder + "/" + str(element["id"]) + ".jpg")

            plt.imshow(img)
            plt.show()
            cont +=1

            enc = False

    i += 1
print(cont)
print(series)
print(collections)
