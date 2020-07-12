# -*- coding: utf-8 -*-
import cv2 as cv
import cv2.cv2 as cvv
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology


#先竖着来
def VThin(image, array):
    h, w = image.shape
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = int(image[i, j - 1]) + int(image[i, j]) + int(image[i, j + 1]) if 0 < j < w - 1 else 1
                if image[i, j] == 255 and M != 255 * 3:#三个连起来是都是255就是中间的点，还轮不到它细化
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 0:
                                a[k * 3 + l] = 1#黑色权重为1,白色权重为0
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = 255 if array[sum] == 0 else 0
                    if array[sum] == 1:
                        NEXT = 0
    return image

# 再横着来
def HThin(image, array):
    h, w = image.shape
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = int(image[i - 1, j]) + int(image[i, j]) + int(image[i + 1, j]) if 0 < i < h - 1 else 1
                if image[i, j] == 255 and M != 255*3:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 0:
                                a[k * 3 + l] = 1#黑色的权重是1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = 255 if array[sum] ==0 else 0
                    if array[sum] == 1:
                        NEXT = 0
    return image


def Xihua(image,array,num=15):
    for i in range(num):
        VThin(image,array)
        HThin(image,array)
    return image

#使用查表法之映射表，一共16*16总256种情况
array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
        1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
        0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
        1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
        1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
        1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
        0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
        1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
        0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
        1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
        1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
        1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
        1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
        1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]



if __name__ == '__main__':
    #读取图像
    ori_src = cv.imread('F:\doc\opencv_test\mor\GuangGu_ori.jpg', 0)
    #step1.灰度图开运算+高斯模糊
    kernel_3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kernel_5 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_cir = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kerel_dilate = np.uint8(np.zeros((5,5)))
    for i in range(5):
        kerel_dilate[2, i] = 1
        kerel_dilate[i, 2] = 1

    OPEN_img3 = cv.morphologyEx(ori_src, cv.MORPH_OPEN, kernel_3)
    Gauss_img = cv.GaussianBlur(OPEN_img3, (5, 5), 0)
   # median1 = cv.medianBlur(ori_src, 5)
    #step2.二值化
    ret, binaryImg = cv.threshold(Gauss_img, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)#这是一种自适应阈值，当图像有两个峰值时比较适用
    print(ret)
    adap_binary = cv.adaptiveThreshold(Gauss_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 2)
    img_bitwiseAnd = cv.bitwise_and(adap_binary,binaryImg)

    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_gauss.jpg',Gauss_img)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_OPEN_img3.jpg', OPEN_img3)
    #cv.imwrite('F:\doc\opencv_test\mor\GuangGu_OPEN_img5.jpg', OPEN_img5)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_img_bitwiseAnd.jpg', img_bitwiseAnd)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_binaryImg.jpg', binaryImg)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_adap_binary.jpg', adap_binary)
    # 使用面积 去除小噪点区域
    contours, hierarch = cv.findContours(img_bitwiseAnd, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < 100:
            cv.drawContours(img_bitwiseAnd, [contours[i]], 0, 0, -1)
    #最后的一步，平滑边缘&细化
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_img_bitwiseAnd2.jpg', img_bitwiseAnd)
    closefirst_img = cv.morphologyEx(img_bitwiseAnd, cv.MORPH_CLOSE, kernel_5)
    openfirst_img = cv.morphologyEx(closefirst_img, cv.MORPH_OPEN, kernel_5)
    median1 = cv.medianBlur(closefirst_img, 5)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_img_bitwiseAndfinal.jpg', img_bitwiseAnd)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_close_img.jpg', closefirst_img)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_openfirst_img.jpg', openfirst_img)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_median1.jpg', median1)
    dst = median1.copy()
    skeletonF = Xihua(dst, array)
    cv.imwrite('F:\doc\opencv_test\mor\GuangGu_skeletonF.jpg', skeletonF)
    cv.waitKey(0)








