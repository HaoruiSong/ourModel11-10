import os
import shutil

path = "D:\\re-ID-dataset\\LTCC_ReID"
clothcot = 0
clothdict = {}

def copyfile(category):
    global clothcot
    global clothdict
    category_path = path + "\\" + category
    for filename in os.listdir(category_path):
        splits = filename.split('_')
        ID = splits[0]
        cloth = splits[1]
        camera = splits[2][1:]
        useless = splits[3]
        if ID + 'c' + cloth in clothdict.keys():
            cloth = clothdict[ID + 'c' + cloth]
        else:
            clothdict[ID + 'c' + cloth] = clothcot
            cloth = clothdict[ID + 'c' + cloth]
            clothcot += 1

        newname = '0' + ID + '_' + 'c' + camera + 's' + str(cloth) + '_' + useless
        frompath = category_path + '\\' + filename
        topath = path + "\\bounding_box_" + category + "\\" + newname
        print(frompath, topath)
        shutil.copyfile(frompath, topath)

copyfile("train")
