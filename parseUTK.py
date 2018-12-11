import os, csv

with open("UTKfacesaligngenderO.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["img_name","gender"])#"age","gender","race","date&time"])
    for path, dirs, files in os.walk("alignedimgs"):
        for filename in files:
            labels = filename.split("_")
            if len(labels)==4:
                print(labels[2])
                labels[3] = labels[3][:-13]
                if int(labels[2]) == 4:
                    writer.writerow([filename]+[labels[1]])
