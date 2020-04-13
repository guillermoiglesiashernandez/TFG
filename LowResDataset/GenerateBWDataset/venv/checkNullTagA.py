import json, glob, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

zeros = np.zeros(93)

for file in glob.glob('MetadataImg/*'):
    print(file)
    with open(file, 'r', encoding="utf8") as f:
        data = json.load(f)
    for element in data:
        if (np.array_equal(np.array(element["tagArray"]), zeros)):
            print(element)