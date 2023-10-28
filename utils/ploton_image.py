import cv2
import os
import argparse
from utils import plot_one_box
from lib_save import Imageprocessing, read_save
import numpy as np
import json
from pathlib import Path
import random



def get_boxes_in_contours(contours, area_filter=0):
    '''

    :param contours:
    :param area_filter:
    :return: contours, box : [[left,top],[right,bottom]]
    '''
    filter_contours = []
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_filter:
            filter_contours.append(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            topleft = box.min(axis=0)
            botright = box.max(axis=0)
            boxes.append([topleft, botright])
    return filter_contours, boxes

def preproc_boxes(boxes):
    np_boxes = np.array(boxes)
    sort_box = sorted(np_boxes, key=lambda s: s[1][0])
    boxes = sort_box[0:-1]
    return boxes

def draw_box(result_im,ori_im,params,delete_mostleft = True):
    # read_params
    image, _, _ = read_save().read_params(params, result_im)
    # print(image["final"])
    # find contour output bounding box
    try:
        _,contours, __ = cv2.findContours(image["final"], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    except:

        contours, __ = cv2.findContours(image["final"], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    _, boxes = get_boxes_in_contours(contours, 50)

    if delete_mostleft:
        boxes = preproc_boxes(boxes)
    # plot bounding box in plot box as yolo
    for box in boxes:
        names = ["burr"]
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # print(colors)
        plot_one_box(box, ori_im, color=(0, 0, 200), label="burr", line_thickness=3)
    return ori_im

def draw_box_by_boxes(to_draw_img,boxes):
    # read_param
    for box in boxes:
        names = ["burr"]
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # print(colors)
        plot_one_box(box, to_draw_img, color=(0, 200, 0), label="burr", line_thickness=3)
    return to_draw_img

def draw_result(opt,delete_mostleft = True):
    # read json params
    # read image result
    result_names = os.listdir(opt.result_path)
    # read original image
    ori_names = os.listdir(opt.ori_path)

    for i in range(len(result_names)):
        result_im = cv2.imread(opt.result_path+"/"+result_names[i])

        ori_im = cv2.imread(opt.ori_path + "/" + ori_names[i])

        # read_params
        image, _, _ = read_save().read_params(params, result_im)
        # print(image["final"])
        # find contour output bounding box
        contours, __ = cv2.findContours(image["final"], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, boxes = get_boxes_in_contours(contours,50)

        if delete_mostleft:
            boxes = preproc_boxes(boxes)

        # plot bounding box in plot box as yolo

        for box in boxes:
            names = ["burr"]
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            # print(colors)
            plot_one_box(box, ori_im,color = (0,0,200),label = "burr", line_thickness = 3)

        cv2.imshow("asd",ori_im)
        cv2.waitKey(0)
        # save
        cv2.imwrite(opt.save_path+"/"+result_names[i],ori_im)

def get_box(result_im,params,delete_mostleft = True):
    # read_params
    image, _, _ = read_save().read_params(params, result_im)
    # print(image["final"])
    # find contour output bounding box
    contours, __ = cv2.findContours(image["final"], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    _, boxes = get_boxes_in_contours(contours, 50)

    if delete_mostleft:
        boxes = preproc_boxes(boxes)
    return boxes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', nargs='+', type=str, default='output/al2_fake/ori',
                        help='output/result_algorithm1/ori')
    parser.add_argument('--result_path', type=str, default='output/al2_fake/result')  # file/folder, 0 for webcam
    parser.add_argument('--params', type=str, default="config/params_plot_label.json", help='inference size (pixels)')
    parser.add_argument('--save_path',type= str,default="output/al2_fake/yolo_label")
    opt = parser.parse_args()
    print(opt)
    try:
        with Path(opt.params).open("r") as f:
            params = json.load(f)

    except:
        print("can't read")

    if not os.path.isdir(opt.save_path):
        os.mkdir(opt.save_path)
    draw_result(opt)