import json, glob

i = 0
enc = False
for myFile in glob.glob('MetadataImgs/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    enc = False
    while enc==False:
        enc = True
        for element in data:
                folder = '%04d' % (int(element["id"]) % 1000)
                try:
                    f = open("C:/Users/Guillermo/Desktop/Imagenes/danbooru-images/danbooru-images/" + folder + "/" + str(element["id"]) + ".jpg")
                except IOError:
                    enc = False
                    data.remove(element)
                finally:
                    f.close()
        print(enc)

    file = open("MetadataImgs/" + str(i) + ".json", "w")
    n = file.write(json.dumps(data, indent=4))
    file.close()
    i += 1