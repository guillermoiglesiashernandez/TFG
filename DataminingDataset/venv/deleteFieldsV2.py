import json, glob

i = 0
for myFile in glob.glob('MetadataImgs/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        del element['rating']
        del element['file_ext']
        del element['is_banned']

    file = open("MetadataImgCleaned/" + str(i) + ".json", "w")
    n = file.write(json.dumps(data, indent=4))
    file.close()
    i += 1