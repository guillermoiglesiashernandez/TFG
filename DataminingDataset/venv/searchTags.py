import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

i = 0
cont = 0

for myFile in glob.glob('MetadataImgCleaned/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        for tag in element['tags']:
            if tag['name'] == "greyscale":
                print(element['id'])
                print(tag['name'])
                folder = '%04d' % (int(element["id"]) % 1000)

                img = mpimg.imread("C:/Users/Guillermo/Desktop/Imagenes/danbooru-images/danbooru-images/" + folder + "/" + str(element["id"]) + ".jpg")

                #plt.imshow(img)
                #plt.show()
                cont +=1
    i += 1

print(cont)