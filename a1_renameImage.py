import imghdr
import os

imageType = ['jpg', 'jpeg', 'bmp', 'dib', 'png', 'tiff', 'rgb', 'tif']

def isImage(path):
    if imghdr.what(path) not in imageType:
        return False
    return True


def renameByNumber(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if isImage(os.path.join(root, file)) == False:
                print("File ", os.path.join(root, file), " is not image!")
                os.remove(os.path.join(root, file))
                continue
            file_new = format(num,'08d') + os.path.splitext(file)[1]
            os.rename(os.path.join(root, file), os.path.join(root, file_new))
            num += 1
            if num % 10000 == 0:
                print(">> ", num, " images have been rename.")
    print("Rename by number complete!")


def addKeyForImage(path, key_ori, key_new):
    index = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_new = file.replace(key_ori, key_new)
            os.rename(os.path.join(root, file), os.path.join(root, file_new))
            index += 1
            if index % 10000 == 0:
                print(">> ", index, " images have been add key.")
    print("Add key for image complete!")


if __name__ == "__main__":
    #renameByNumber("./data/train")
    addKeyForImage("./data/train/gou", ".", "_gou_.")
    addKeyForImage("./data/train/hua", ".", "_hua_.")
    addKeyForImage("./data/train/ji", ".", "_ji_.")
    addKeyForImage("./data/train/ma", ".", "_ma_.")
    addKeyForImage("./data/train/mao", ".", "_mao_.")
    addKeyForImage("./data/train/niu", ".", "_niu_.")
    addKeyForImage("./data/train/songshu", ".", "_songshu_.")
    addKeyForImage("./data/train/xiang", ".", "_xiang_.")
    addKeyForImage("./data/train/yang", ".", "_yang_.")
    addKeyForImage("./data/train/zhizhu", ".", "_zhizhu_.")

