import os
import cv2
import numpy as np

def listDir(dir):
    """
    Function Name: listDir

    Description: Input directory and return list of all name in that directory

    Argument:
        dir [string] -> [directory]

    Return:
        [list] -> [name of all files in the directory]

    Edited by: 12-4-2020 [Pawat]
    """
    fileNames = os.listdir(dir)
    Name = []
    for fileName in fileNames:
        Name.append(fileName)
        # print(fileName)

    return Name



def contour_to_box(contour):
    """
    Function Name: __contour_to_box

    Description: [summary]

    Argument:
        contour [] -> [sub contour from opencv]

    Return:
        [list] -> [topleft, botright]

    Edited by: [date] [author name]
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    topleft = box.min(axis=0)
    botright = box.max(axis=0)
    # print(topleft)
    # print(botright)
    return [topleft, botright]


def imshow_fit(window_name, im):
    # print(im)
    if max(im.shape) <= 1800:
        resize_factor = 2
        im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)))

    elif max(im.shape) > 1800 and max(im.shape) <= 4500:
        resize_factor = 5
        im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)))
    elif max(im.shape) > 4500 and max(im.shape) <= 8000:
        resize_factor = 7
        im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)))
    elif max(im.shape) > 8000:
        resize_factor = 9
        im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)))
    else:
        resize_factor = 1
    # cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,im)