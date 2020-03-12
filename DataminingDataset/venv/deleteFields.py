import json, glob

i = 0
for myFile in glob.glob('MetadataStructured/*'):
    print(myFile)
    with open(myFile, 'r', encoding="utf8") as f:
        data = json.load(f)

    for element in data:
        del element['created_at']
        del element['uploader_id']
        del element['score']
        del element['source']
        del element['md5']
        del element['last_commented_at']
        del element['is_note_locked']
        del element['last_noted_at']
        del element['is_rating_locked']
        del element['approver_id']
        del element['file_size']
        del element['is_status_locked']
        del element['up_score']
        del element['down_score']
        del element['is_pending']
        del element['is_flagged']
        del element['is_deleted']
        del element['updated_at']
        del element['pixiv_id']
        del element['favs']

    file = open("MetadataCleaned/" + str(i) + ".json", "w")
    n = file.write(json.dumps(data, indent=4))
    file.close()
    i += 1