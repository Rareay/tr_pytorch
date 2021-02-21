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


def chooseFiles(path, chooseNums):
    newDir = "./testData/"
    namelist = findFile(path)
    random.seed(100)
    obj_namelist = random.sample(namelist, chooseNums)
    for i, name in enumerate(obj_namelist):
        moveFile(name, newDir + name)
        if (i + 1) % 1000 == 0:
            print(i + 1, " images have been processed.")
    print("complete: ", path)


if __name__ == "__main__":
    chooseFiles("data/train/gou", 500)
    chooseFiles("data/train/hua", 500)
    chooseFiles("data/train/ji", 500)
    chooseFiles("data/train/ma", 500)
    chooseFiles("data/train/mao", 500)
    chooseFiles("data/train/niu", 500)
    chooseFiles("data/train/songshu", 500)
    chooseFiles("data/train/xiang", 500)
    chooseFiles("data/train/yang", 500)
    chooseFiles("data/train/zhizhu", 500)

