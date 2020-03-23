import glob, os
from PIL import Image

for folder in glob.glob("C:/Users/Guillermo/Desktop/TFG/DatasetBW/BWImages/*"):
    print(folder)
    folderName = os.path.basename(folder)

    os.makedirs("C:/Users/Guillermo/Desktop/TFG/DatasetBW/LowResImages/" + folderName)

    for img in glob.glob(folder + "/*.jpg"):
        name = os.path.basename(img)

        img = Image.open(img)
        img = img.resize((128,128),Image.ANTIALIAS)
        img.save("C:/Users/Guillermo/Desktop/TFG/DatasetBW/LowResImages/" + folderName + "/" + name)