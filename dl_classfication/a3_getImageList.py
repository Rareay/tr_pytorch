import os
import re

# 文件名的数字表示类别，如：300～400表示一类，400～500表示一类
def findfile_1(img_path, filename):
    file_write = open(filename, "w")
    line = ""
    for root,dirs,files in os.walk(img_path):
        for file in files:
            name = file.split('.',1)
            if name[1] == "jpg":
                if int(name[0]) >= 300 and int(name[0]) < 400:
                    line = line + os.path.join(root, file) + " 0\n"
                elif int(name[0]) >= 400 and int(name[0]) < 500:
                    line = line + os.path.join(root, file) + " 1\n"
                elif int(name[0]) >= 500 and int(name[0]) < 600:
                    line = line + os.path.join(root, file) + " 2\n"
                elif int(name[0]) >= 600 and int(name[0]) < 700:
                    line = line + os.path.join(root, file) + " 3\n"
                elif int(name[0]) >= 700 and int(name[0]) < 800:
                    line = line + os.path.join(root, file) + " 4\n"
    file_write.write(line)


def findfile(labels, img_path, filename):
    if os.path.exists(filename):
        os.remove(filename)
    file_write = open(filename, "a+")
    #file_write = open(filename, "w")
    line = ""
    file_nums = 0
    for root, dirs, files in os.walk(img_path):
        for file in files:
            tag_index = 0
            for tag in labels:
                if re.search(tag, file) != None:
                    file_write.write(os.path.join(root, file) + " " + str(tag_index) + "\n")
                    #line = line + os.path.join(root, file) + " " + str(tag_index) + "\n"
                    file_nums += 1
                    break
                tag_index += 1
    #file_write.write(line)
    print(img_path, " : have ", file_nums, " iamges.")

label1 = [
          "_gou_",
          "_hua_",
          "_ji_",
          "_ma_",
          "_mao_",
          "_niu_",
          "_songshu_",
          "_xiang_",
          "_yang_",
          "_zhizhu_",
          ]
label2 = ["_e_", "_r_", "_g_", "_y_"]
label3 = ['cat', "dog"]
lables = label3

def getClass():
    return lables

if __name__ == '__main__':
    # 使用相对路径
    #findfile(lables, r"data/tlc/color/val", "./imagelist/val.txt" )
    #findfile(lables, r"data/tlc/color/train", "./imagelist/train.txt")
    #findfile(lables, r"./data/raw-img/test", "./imagelist/val.txt" )
    #findfile(lables, r"./data/raw-img/train", "./imagelist/train.txt")
    findfile(lables, r"./data/dogvscat/train", "./imagelist/train.txt")
    findfile(lables, r"./data/dogvscat/valide", "./imagelist/val.txt")
