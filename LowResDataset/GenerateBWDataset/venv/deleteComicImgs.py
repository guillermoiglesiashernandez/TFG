import json, glob, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

cont = 0

for file in glob.glob('MetadataImg/*'):
    print(file)
    with open(file, 'r', encoding="utf8") as f:
        data = json.load(f)

    delete = [("BWImages/" + ('%04d' % (int(img["id"]) % 1000)) + "/" + str(img["id"]) + ".jpg") for img in data for tag in img["tags"] if (tag["name"] == "comic")]

    for img in delete:
        try:
            open(img)
            os.remove(img)
        except FileNotFoundError:
            print("")
        finally:
            print("")
        cont += 1

print("Imagenes de comic: " + str(cont))