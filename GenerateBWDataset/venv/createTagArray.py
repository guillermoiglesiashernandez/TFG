import json, glob, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

for file in glob.glob('MetadataImg/*'):
    print(file)
    with open(file, 'r', encoding="utf8") as f:
        data = json.load(f)
    for element in data:
        tagArray = np.zeros(93)
        for tag in element["tags"]:
            tagName = tag["name"]
            if tagName=="1girl":
                tagArray[0] = 1
            elif tagName=="solo":
                tagArray[0] = 1
            elif tagName == "long_hair":
                tagArray[1] = 1
            elif tagName=="smile":
                tagArray[2] = 1
            elif tagName == "short_hair":
                tagArray[3] = 1
            elif tagName=="multiple_girls":
                tagArray[4] = 1
            elif tagName == "2girls":
                tagArray[4] = 1
            elif tagName=="3girls":
                tagArray[4] = 1
            elif tagName == "blush":
                tagArray[5] = 1
            elif tagName=="open_mouth":
                tagArray[6] = 1
            elif tagName == ":d":
                tagArray[6] = 1
            elif tagName=="blonde_hair":
                tagArray[7] = 1
            elif tagName == "breasts":
                tagArray[8] = 1
            elif tagName=="blue_eyes":
                tagArray[9] = 1
            elif tagName == "skirt":
                tagArray[10] = 1
            elif tagName=="pleated_skirt":
                tagArray[10] = 1
            elif tagName == "hat":
                tagArray[11] = 1
            elif tagName=="brown_hair":
                tagArray[12] = 1
            elif tagName == "red_eyes":
                tagArray[13] = 1
            elif tagName=="black_hair":
                tagArray[14] = 1
            elif tagName == "dress":
                tagArray[15] = 1
            elif tagName=="ribbon":
                tagArray[16] = 1
            elif tagName == "hair_ribbon":
                tagArray[16] = 1
            elif tagName=="thighhighs":
                tagArray[17] = 1
            elif tagName == "hair_ornament":
                tagArray[18] = 1
            elif tagName=="hairclip":
                tagArray[18] = 1
            elif tagName == "gloves":
                tagArray[19] = 1
            elif tagName=="elbow_gloves":
                tagArray[19] = 1
            elif tagName == "bow":
                tagArray[20] = 1
            elif tagName=="school_uniform":
                tagArray[21] = 1
            elif tagName == "serafuku":
                tagArray[21] = 1
            elif tagName=="1boy":
                tagArray[22] = 1
            elif tagName == "simple_background":
                tagArray[23] = 1
            elif tagName=="twintails":
                tagArray[24] = 1
            elif tagName == "brown_eyes":
                tagArray[25] = 1
            elif tagName=="green_eyes":
                tagArray[26] = 1
            elif tagName == "blue_hair":
                tagArray[27] = 1
            elif tagName=="large_breasts":
                tagArray[28] = 1
            elif tagName == "sitting":
                tagArray[29] = 1
            elif tagName=="animal_ears":
                tagArray[30] = 1
            elif tagName == "white_background":
                tagArray[31] = 1
            elif tagName=="weapon":
                tagArray[32] = 1
            elif tagName == "long_sleeves":
                tagArray[33] = 1
            elif tagName=="cleavage":
                tagArray[34] = 1
            elif tagName == "closed_eyes":
                tagArray[35] = 1
            elif tagName=="jewelry":
                tagArray[36] = 1
            elif tagName == "shirt":
                tagArray[37] = 1
            elif tagName=="bangs":
                tagArray[38] = 1
            elif tagName == "medium_breasts":
                tagArray[39] = 1
            elif tagName=="very_long_hair":
                tagArray[40] = 1
            elif tagName == "purple_eyes":
                tagArray[41] = 1
            elif tagName=="bare_shoulders":
                tagArray[42] = 1
            elif tagName == "ponytail":
                tagArray[43] = 1
            elif tagName=="hair_bow":
                tagArray[44] = 1
            elif tagName == "flower":
                tagArray[45] = 1
            elif tagName=="purple_hair":
                tagArray[46] = 1
            elif tagName == "pink_hair":
                tagArray[47] = 1
            elif tagName=="navel":
                tagArray[48] = 1
            elif tagName == "midriff":
                tagArray[48] = 1
            elif tagName=="yellow_eyes":
                tagArray[49] = 1
            elif tagName == "black_legwear":
                tagArray[50] = 1
            elif tagName=="silver_hair":
                tagArray[51] = 1
            elif tagName == "wings":
                tagArray[52] = 1
            elif tagName=="braid":
                tagArray[53] = 1
            elif tagName == "tail":
                tagArray[54] = 1
            elif tagName=="boots":
                tagArray[55] = 1
            elif tagName == "hairband":
                tagArray[56] = 1
            elif tagName=="pantyhose":
                tagArray[57] = 1
            elif tagName == "panties":
                tagArray[57] = 1
            elif tagName=="green_hair":
                tagArray[58] = 1
            elif tagName == "underwear":
                tagArray[59] = 1
            elif tagName=="ahoge":
                tagArray[60] = 1
            elif tagName == "holding":
                tagArray[61] = 1
            elif tagName=="japanese_clothes":
                tagArray[62] = 1
            elif tagName == "full_body":
                tagArray[63] = 1
            elif tagName=="food":
                tagArray[64] = 1
            elif tagName == "red_hair":
                tagArray[65] = 1
            elif tagName=="detached_sleeves":
                tagArray[66] = 1
            elif tagName == "swimsuit":
                tagArray[67] = 1
            elif tagName=="glasses":
                tagArray[68] = 1
            elif tagName == "standing":
                tagArray[69] = 1
            elif tagName=="multiple_boys":
                tagArray[70] = 1
            elif tagName == "2boys":
                tagArray[70] = 1
            elif tagName=="one_eye_closed":
                tagArray[71] = 1
            elif tagName == "necktie":
                tagArray[72] = 1
            elif tagName=="heart":
                tagArray[73] = 1
            elif tagName == "sword":
                tagArray[74] = 1
            elif tagName=="jacket":
                tagArray[75] = 1
            elif tagName == "short_sleeves":
                tagArray[76] = 1
            elif tagName=="shoes":
                tagArray[77] = 1
            elif tagName == "eyebrows_visible_through_hair":
                tagArray[78] = 1
            elif tagName=="upper_body":
                tagArray[79] = 1
            elif tagName == "sky":
                tagArray[80] = 1
            elif tagName=="white_hair":
                tagArray[81] = 1
            elif tagName == "barefoot":
                tagArray[82] = 1
            elif tagName=="collarbone":
                tagArray[83] = 1
            elif tagName == "frills":
                tagArray[84] = 1
            elif tagName=="hair_between_eyes":
                tagArray[85] = 1
            elif tagName == "earrings":
                tagArray[86] = 1
            elif tagName=="white_legwear":
                tagArray[87] = 1
            elif tagName == "shorts":
                tagArray[88] = 1
            elif tagName=="day":
                tagArray[89] = 1
            elif tagName == "closed_mouth":
                tagArray[90] = 1
            elif tagName=="cloud":
                tagArray[91] = 1
            elif tagName == "belt":
                tagArray[92] = 1

        tagArray = tagArray.tolist()
        element.update({"tagArray":tagArray})

    file = open(file, "w")
    n = file.write(json.dumps(data, indent=4))
    file.close()