import os, shutil
import random


def moveFile(src, dst):
    if not os.path.isfile(src):
        print("%s not exits!" % (src))
    else:
        fpath, fname = os.path.split(dst)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(src, dst)


def copyFile(src, dst):
    if not os.path.isfile(src):
        print("%s not exits!" % (src))
    else:
        fpath, fname = os.path.split(dst)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(src, dst)


def findFile(path):
    namelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            namelist.append(os.path.join(root, file))
    return namelist


def chooseFiles(path, chooseNums, out_dir):
    newDir = out_dir
    namelist = findFile(path)
    random.seed(100)
    obj_namelist = random.sample(namelist, chooseNums)
    for i, name in enumerate(obj_namelist):
        moveFile(name, newDir + name)
        if (i + 1) % 1000 == 0:
            print(i + 1, " images have been processed.")
    print("complete: ", path)


if __name__ == "__main__":
    #out_dir = "./testData/"
    out_dir = "./valData/"
    #chooseFiles("data/tlc/color/train/e", 1000, out_dir)
    #chooseFiles("data/tlc/color/train/r", 1000, out_dir)
    #chooseFiles("data/tlc/color/train/g", 1000, out_dir)
    #chooseFiles("data/tlc/color/train/y", 1000, out_dir)
    chooseFiles("data/dogvscat/train/dog", 2500, out_dir)

