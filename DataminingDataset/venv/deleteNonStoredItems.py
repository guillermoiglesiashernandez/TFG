import json, glob

i=0
for myFile in glob.glob('MetadataCleaned/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    delete = [obj for obj in data if (int(obj["id"]) % 1000 > 150)]
    for item in delete:
        data.remove(item)

    file = open("MetadataStored/" + str(i) + ".json", "w")
    n = file.write(json.dumps(data, indent=4))
    file.close()
    i += 1