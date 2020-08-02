import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

i = 0
cont = 0
error = 0
wallcomic = 0

for myFile in glob.glob('C:/Users/Guillermo/Desktop/TFG/Archivos Grandes/Metadata Transitorio/MetadataImgCleaned/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        cont += 1
        if int(element['image_width']) != 0 and int(element['image_height']) != 0:
            if (int(element['image_width']) / int(element['image_height']) >= 4) or (int(element['image_height']) / int(element['image_width']) >= 4):
                folder = '%04d' % (int(element["id"]) % 1000)
                img = mpimg.imread("C:/Users/Guillermo/Desktop/TFG/Archivos Grandes/Dataset/danbooru-images/danbooru-images/" + folder + "/" + str(element["id"]) + ".jpg")

                print(element['id'])
                plt.imshow(img)
                plt.show()
                wallcomic +=1

    i += 1

print("Contador: " + str(cont))
print("Imagenes demasiado altas/anchas: " + str(wallcomic))
print("Errores: " + str(error))