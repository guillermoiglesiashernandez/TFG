import json, glob

i=0
for myFile in glob.glob('MetadataRaw/*.json'):
  print(myFile)
  with open(myFile, 'r', encoding="utf8") as f:
    data = json.load(f)

  file = open("MetadataStructured/" + str(i) + ".json", "w")
  n = file.write(json.dumps(data, indent=4))
  file.close()
  i += 1