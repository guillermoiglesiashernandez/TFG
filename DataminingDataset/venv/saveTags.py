import json, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

i = 0
cont = []
tagList = []
for myFile in glob.glob('MetadataImgCleaned/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        for tag in element['tags']:
            if tag in tagList:
                cont[tagList.index(tag)]+=1
            else:
                tagList.append(tag)
                cont.append(0)

    salida1 = open("Tags/Tags.txt", 'w', encoding="utf8")
    salida1.write(json.dumps(tagList, indent=4))

    salida2 = open("Tags/Values.txt", 'w', encoding="utf8")
    salida2.write(json.dumps(cont, indent=4))

    i += 1
