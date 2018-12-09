import os, csv

with open("UTKfacesalignrace.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["img_name","race"])#"age","gender","race","date&time"])
    for path, dirs, files in os.walk("alignedimgs"):
        for filename in files:
            labels = filename.split("_")
            if len(labels)==4:
                print(labels[2])
                labels[3] = labels[3][:-13]
                writer.writerow([filename]+[labels[2]])
