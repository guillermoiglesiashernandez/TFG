import os, zipfile, glob
from datetime import datetime

print("-----GUARDANDO BACKUP DEL ENTRENO-----")
fileName = "resultsBk" + datetime.now().strftime("%d%m%Y-%H%M%S") + ".zip"
with zipfile.ZipFile(fileName, "w") as zip:
    zip.write("confMatrix/")
    files = glob.glob("confMatrix/*")
    for file in files:
        zip.write(file)
        os.remove(file)

    files = glob.glob("performance/trainPerformance/*.png")
    for file in files:
        zip.write(file)
        os.remove(file)
    files = glob.glob("performance/trainPerformance/performance/*.txt")
    for file in files:
        zip.write(file)
        os.remove(file)
    files = glob.glob("performance/trainPerformance/results/*.png")
    for file in files:
        zip.write(file)
        os.remove(file)

    files = glob.glob("performance/testPerformance/*.png")
    for file in files:
        zip.write(file)
        os.remove(file)
    files = glob.glob("performance/testPerformance/performance/*.txt")
    for file in files:
        zip.write(file)
        os.remove(file)
    files = glob.glob("performance/testPerformance/results/*.png")
    for file in files:
        zip.write(file)
        os.remove(file)

    files = glob.glob("performance/epochPerformance/*.png")
    for file in files:
        zip.write(file)
        os.remove(file)
    files = glob.glob("performance/epochPerformance/performance/*.txt")
    for file in files:
        zip.write(file)
        os.remove(file)
    files = glob.glob("performance/epochPerformance/results/*.png")
    for file in files:
        zip.write(file)
        os.remove(file)

    for file in glob.glob("saved_model/*"):
        zip.write(file)