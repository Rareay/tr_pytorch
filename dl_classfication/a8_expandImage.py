import cv2
import numpy as np
import random
import os


def findFile(path):
    '''
    return namelist
    '''
    namelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            namelist.append(os.path.join(root, file))
    return namelist


# 平移
def doTranslation(img, w_rate, h_rate):
    '''
    :param w_off: -0.15  -  +0.15
    :param h_off: -0.15  -  +0.15
    '''
    (h, w) = img.shape[:2]
    w_off = int(w_rate * w)
    h_off = int(h_rate * h)
    M = np.float32([[1, 0, w_off], [0, 1, h_off]])
    dst = cv2.warpAffine(img, M, (w, h))
    return dst

# 缩放
def doResize(img, w_rate, h_rate, deviation):
    '''
    have some problem!
    :param w_rate:
    :param h_rate:
    '''
    (h, w) = img.shape[:2]
    w_dst = int((w_rate + 1) * w)
    h_dst = int((h_rate + 1) * h)
    final_matrix = np.zeros(img.shape, np.uint8)
    x1 = 0
    y1 = h
    x2 = w
    y2 = 0
    dst = cv2.resize(img, (w_dst, h_dst))
    if h < h_dst:
        final_matrix[y2:y1, x1:x2] = dst[y2+deviation:y1+deviation, x1+deviation:x2+deviation]
    else:
        if deviation == 0:
            final_matrix[y2:h_dst, x1:w_dst] = dst[y2:h_dst, x1:w_dst:]
        else:
            final_matrix[y2+deviation:h_dst+deviation, x1+deviation:w_dst+deviation] = dst[y2+h_dst, x1:w_dst]
    return final_matrix


# 翻转
def doFlip(img, direction):
    '''
    :param direction: 0 / 1  / -1
    '''
    out = img.copy()
    cv2.flip(img, direction, out)
    return  out



# 旋转
def doRotation(img, ra):
    '''
    :param ra: 0 - 360
    '''
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, ra, 1.0)
    dst = cv2.warpAffine(img, M, (w, h))
    return dst
 
# 剪切
def doCut(img, top_rate=0., botton_rate=0., left_rate=0., right_rate=0.):
    '''
    :param top_rate:     0 - 0.2
    :param bottom_rate:  0 - 0.2
    :param left_rate:    0 - 0.2
    :param right_rate:   0 - 0.2
    '''
    (h, w) = img.shape[:2]
    top_off = int(h * top_rate)
    botton_off = int(h * botton_rate)
    left_off = int(w * left_rate)
    right_off = int(w * right_rate)
    x1 = 0
    y1 = h
    x2 = w
    y2 = 0
    final_matrix = np.zeros(img.shape, np.uint8)
    final_matrix[y2 + top_off: y1 - botton_off, x1 + left_off:x2-right_off] = img[y2 + top_off:y1 - botton_off, x1 + left_off:x2 - right_off]

    return final_matrix



# 噪声
def doNoiseJiaoyan(img, prob):
    '''
    :param prob: 0.001  -  0.01
    '''
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


def doNoiseGasuss(img, mean=0.5, var=0.02):
    '''
    :param mean: 0.1  -  0.5
    :param var:  mean/20  -  mean/10
    '''
    img = img / 255
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

# 模糊
def doGasussBlur(img, sigma):
    '''
    :param kernel: (5,5)
    :param sigma:  1.  -  3.
    '''
    out = cv2.GaussianBlur(img, (5,5), sigma)
    return out


def batchTranslation(namelist, save_dir, expand_num=0):
    base_dir = save_dir + "/translation/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    filename_num = 0
    obj_num = expand_num
    for _ in range(obj_num):
        index_rand = int(random.random() * (len(namelist) - 1))
        filename = namelist[index_rand]
        img = cv2.imread(filename)
        w_rate = (random.random() - 0.5) * 1.5 / 5
        h_rate = (random.random() - 0.5) * 1.5 / 5
        img2 = doTranslation(img, w_rate, h_rate)
        just_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = base_dir + just_name + "_" + format(filename_num, '04d') + ".jpg"
        cv2.imwrite(new_filename, img2)
        filename_num += 1


def batchFlip(namelist, save_dir, expand_num=0):
    base_dir = save_dir + "/flip/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    filename_num = 0
    obj_num = expand_num
    for _ in range(obj_num):
        index_rand = int(random.random() * (len(namelist) - 1))
        filename = namelist[index_rand]
        img = cv2.imread(filename)
        direction = int(random.random() * 3) - 1
        img2 = doFlip(img, direction)
        just_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = base_dir + just_name + "_" + format(filename_num, '04d') + ".jpg"
        cv2.imwrite(new_filename, img2)
        filename_num += 1


def batchRotation(namelist, save_dir, expand_num=0):
    base_dir = save_dir + "/rotation/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    filename_num = 0
    obj_num = expand_num
    for _ in range(obj_num):
        index_rand = int(random.random() * (len(namelist) - 1))
        filename = namelist[index_rand]
        img = cv2.imread(filename)
        rota = random.random() * 360
        img2 = doRotation(img, rota)
        just_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = base_dir + just_name + "_" + format(filename_num, '04d') + ".jpg"
        cv2.imwrite(new_filename, img2)
        filename_num += 1

def batchCut(namelist, save_dir, expand_num=0):
    base_dir = save_dir + "/cut/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    filename_num = 0
    obj_num = expand_num
    for _ in range(obj_num):
        index_rand = int(random.random() * (len(namelist) - 1))
        filename = namelist[index_rand]
        img = cv2.imread(filename)
        rate = random.random() * 0.2
        img2 = doCut(img, rate, rate, rate, rate)
        just_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = base_dir + just_name + "_" + format(filename_num, '04d') + ".jpg"
        cv2.imwrite(new_filename, img2)
        filename_num += 1



def batchNoiseJiaoyan(namelist, save_dir, expand_num=0):
    base_dir = save_dir + "/noise_jiaoyan/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    filename_num = 0
    obj_num = expand_num
    for _ in range(obj_num):
        index_rand = int(random.random() * (len(namelist) - 1))
        filename = namelist[index_rand]
        img = cv2.imread(filename)
        rate = 0.001 + 0.099 * random.random()
        img2 = doNoiseJiaoyan(img, rate)
        just_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = base_dir + just_name + "_" + format(filename_num, '04d') + ".jpg"
        cv2.imwrite(new_filename, img2)
        filename_num += 1


def batchNoiseGasuss(namelist, save_dir, expand_num=0):
    base_dir = save_dir + "/noise_gasuss/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    filename_num = 0
    obj_num = expand_num
    for _ in range(obj_num):
        index_rand = int(random.random() * (len(namelist) - 1))
        filename = namelist[index_rand]
        img = cv2.imread(filename)
        mean = 0.1 + 0.4 * random.random()
        var = mean / (10 + 10 * random.random())
        img2 = doNoiseGasuss(img, mean, var)
        just_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = base_dir + just_name + "_" + format(filename_num, '04d') + ".jpg"
        cv2.imwrite(new_filename, img2)
        filename_num += 1


def batchGasussBlur(namelist, save_dir, expand_num=0):
    base_dir = save_dir + "/gasuss/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    filename_num = 0
    obj_num = expand_num
    for _ in range(obj_num):
        index_rand = int(random.random() * (len(namelist) - 1))
        filename = namelist[index_rand]
        img = cv2.imread(filename)
        sigma = 0.5 + 4 * random.random()
        img2 = doGasussBlur(img, sigma)
        just_name = os.path.splitext(os.path.basename(filename))[0]
        new_filename = base_dir + just_name + "_" + format(filename_num, '04d') + ".jpg"
        cv2.imwrite(new_filename, img2)
        filename_num += 1


def expandImage(image_dir, category, save_dir='./expandData', nums=0):
    translate = int(nums * 0.15)
    flip = int(nums * 0.1)
    rotation = int(nums * 0.2)
    cut = int(nums * 0.15)
    noise_jiaoyan = int(nums * 0.1)
    noise_gasuss = int(nums * 0.1)
    gasussblur = int(nums * 0.2)
    save_dir = save_dir + "/" + category
    namelist = findFile(image_dir)
    batchTranslation(namelist, save_dir, translate)
    batchFlip(namelist, save_dir, flip)
    batchRotation(namelist, save_dir, rotation)
    batchCut(namelist, save_dir, cut)
    batchNoiseJiaoyan(namelist, save_dir, noise_jiaoyan)
    batchNoiseGasuss(namelist, save_dir, noise_gasuss)
    batchGasussBlur(namelist, save_dir, gasussblur)

    print("Expand %d images, save in '%s'" %(
            translate + flip + rotation + cut + noise_jiaoyan + noise_gasuss + gasussblur,
            save_dir))


if __name__ == "__main__":
    #expandImage('data/raw-img/train/', 'y', nums=5000 - 2642)
    expandImage('data/raw-img/train/hua', 'hua', nums=5000 - 1612)
    expandImage('data/raw-img/train/ji', 'ji', nums=5000 - 2598)
    expandImage('data/raw-img/train/ma', 'ma', nums=5000 - 2123)
    expandImage('data/raw-img/train/mao', 'mao', nums=5000 - 1168)
    expandImage('data/raw-img/train/niu', 'niu', nums=5000 - 1366)
    expandImage('data/raw-img/train/songshu', 'songshu', nums=5000 - 432)
    expandImage('data/raw-img/train/xiang', 'xiang', nums=5000 - 822)
    expandImage('data/raw-img/train/yang', 'yang', nums=5000 - 1320)
    expandImage('data/raw-img/train/zhizhu', 'zhizhu', nums=5000 - 4321)


