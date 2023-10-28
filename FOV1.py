import cv2
import os
import json
import numpy as np
import argparse
import random as rng
import pandas as pd
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
from easydict import EasyDict
from pypylon import pylon
from pathlib import Path
from lib_save import Imageprocessing, read_save
from utils import *
# import array
import math
from utils.ploton_image import draw_box
from copy import deepcopy
# from main_fov22 import Fov2
from statistics import mean


EXTEND_CIRCLE_FACTOR = 0.035
LEFT_BOUND_WARP_IMG = 0.93
# percent of image of left coord of crop sliding window
INITIAL_THRESHOLD = 0.1
INITIAL_PIXEL = 1
CROP_CENTERIMAGE = 0

class Fov1():

    def __init__(self):
        """
            Function Name: __init

            Description: inspect FOV1
                        1. read parameters from comfig main.json
                        2. exceute FOV1 inspection


            Edited by: [2021/8/28] [Pawat]
            """

        # super(Fov2, self).__init__()
        self.log = Log("main.py", stream_level="INFO", stream_enable=True, record_level="WARNING", record_path='log.csv')
        self.log.show("===== start program =====", "INFO")

        self.reading = read_save()

        self.params = {}
        self.kernel_params = {}
        self.opt = {}

        self.imgproc = Imageprocessing()
        self.reading = read_save()

        try:
            with Path("config/main.json").open("r") as f:
                self.opt = json.load(f)
                print(self.opt)
        except:
            self.log.show("config/main.json" + " doesn't exist. Please, select the file again!", "ERROR")

        if self.opt != {}:

            self.opt = EasyDict(self.opt)
            # read params
            try:
                with Path(self.opt.basic.path_params).open("r") as f:
                    self.log.show("read params.json", "DEBUG")
                    self.params = json.load(f)
                    # print("params",params)
            except:
                self.log.show(self.opt.basic.path_params + " doesn't exist. Please, select the file again!", "ERROR")

            # read kernel_params
            try:
                with Path(self.opt.basic.path_kernel).open("r") as f:
                    self.kernel_params = json.load(f)
                    # print(self.rect_params)
            except:
                self.log.show(self.opt.basic.path_kernel + " doesn't exist. Please, select the file again!", "ERROR")

            # params detect red for writing box
            try:
                with Path(self.opt.basic.write_box).open("r") as f:
                    self.params_box = json.load(f)

            except:
                print("can't read")
        if not os.path.isdir("output/al" + str(self.opt.algorithm) + "/"):
            os.mkdir("output/al" + str(self.opt.algorithm) + "/")
        if not os.path.isdir("output/al" + str(self.opt.algorithm) + "/" + "test_test" + "/"):
            os.mkdir("output/al" + str(self.opt.algorithm) + "/" + "test_test" + "/")

            # if self.params != {} and self.kernel_params != {}:
            #     self.main()

            # else:
            #     self.log.show("missing params-json file or rects-json file ", "WARNING")

    @staticmethod
    def __contour_to_box(contour):
        """
        Function Name: __contour_to_box

        Description: [summary]

        Argument:
            contour [] -> [sub contour from opencv]

        Return:
            [list] -> [topleft, botright] or [[x0,y0],[x1,y1]]

        Edited by: [2021/1/28] [Pawat]
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        topleft = box.min(axis=0)
        botright = box.max(axis=0)
        return [topleft, botright]

    def select_closest_contour(self, boxes, reference):  # array of boxes and center point of ref
        """
        Function Name: find_closest

        Description: [summary]

        Argument:
            boxes [list] -> [list all of boxes of detected contours]
            reference [tuplr or array] -> [reference point to compare or find cloest from boxes]

        Parameters:

        Return:
            index [int] -> [index or order of the cloest box or nothing incase there is no close box in threshlod dist]

        Edited by: [12-4-2020] [Pawat]
        """
        lengths = []
        x_lengths = []
        y_lengths = []
        index = None

        if boxes != []:

            for i, box in enumerate(boxes):
                x0, y0 = box[0]
                x1, y1 = box[1]
                center = [(x0 + x1) / 2, (y0 + y1) / 2]
                lengthX = (center[0] - reference[0]) ** 2
                lengthY = (center[1] - reference[1]) ** 2
                length = math.sqrt(lengthY + lengthX)
                x_length = math.sqrt(lengthX)
                y_length = math.sqrt(lengthY)
                lengths.append(length)
                x_lengths.append([i, x_length])
                y_lengths.append([i, y_length])

            index = None
            if lengths != []:

                lengths = np.array(lengths)
                min_length = np.min(lengths)

                if min_length <= 999:  # threshold

                    for i, length in enumerate(lengths):
                        if length == min_length:
                            index = i
                            break

        return index

    def reverse_warp(self,ori_image,warped_image,circles):
        '''
        reverse warp polar to be circular image
        :param ori_image:
        :param warped_image:
        :param circles:
        :return:
        '''
        if circles is not None:
            circle = circles[-1] # -1 means biggest one (it's sorted outside before in this function)
            center = (circle[0], circle[1])
            radius = circle[-1] + int(circle[-1] * (EXTEND_CIRCLE_FACTOR)) # extend circle
        reversed_img = cv2.warpPolar(warped_image, (ori_image.shape[1], ori_image.shape[0]), center, radius, flags=(cv2.WARP_INVERSE_MAP))
        return reversed_img

    def contours_to_boxes(self, contours):
        """
        Function Name: contours_to_boxes

        Description: convert set of contours to boxes

        Argument:
            contours [[arr]] -> contours from cv2.findContours()

        Return:
            boxes[list] -> [topleft, botright] or [[x0,y0],[x1,y1]] converted boxes from contours

        Edited by: [2021/1/28] [Pawat]
        """
        boxes = []
        for contour in contours:
            box = self.__contour_to_box(contour)
            boxes.append(box)
        return boxes

    def select_direction_contour(self,filter_contours, boxes,direction = "low"):
        '''
        to select most cloest direction . . .. [low, left, right, top]
        :param filter_contours:
        :param boxes:
        :param direction:
        :return:
        '''
        # print(len(boxes))
        index = []
        if direction == "low":
            a1 = 1
            a2 = 1
            order = [-1,-2,-3]
        elif direction == "left":
            a1 = 0
            a2 = 0
            order = [0,1,2]
        elif direction == "right":
            a1 = 1
            a2 = 0
            order = [-1,-2,-3]
        elif direction == "top":
            a1 = 0
            a2 = 1
            order = [0,1,2]
        sort_box = sorted(boxes, key=lambda s: s[a1][a2])
        for num in range(int(np.clip(len(order),1,len(boxes)))):
            for i,box in enumerate(boxes):
                if np.all(np.array(box) == np.array(sort_box[order[num]])):
                    index.append(i)
        return index

    def __get_x_y_from_contour(self,contour):
        '''
        To collext data x,y coordinate from the points in contours
        get all x coords and all y corrds from contour
        :param contour:
        :return: list of all x , list of all y coords
        '''
        x = []
        y = []
        for coord in contour:
            x.append(coord[0][0])
            y.append(coord[0][1])
        return x, y

    def crop_warp(self, warp_img, start_deg, end_deg):
        '''
        [WARP IN SPECIFY ANGLE]
        this function for fov2
        crop warp in the image normally 30 to 150
        0 angle is on left side
        :param warp_img:
        :param start_deg:
        :param end_deg:
        :return:
        '''
        y = warp_img.shape[0]
        start_y = int(start_deg * y / 360)
        end_y = int(end_deg * y / 360)
        crop_warp_img = warp_img[start_y:end_y, 0:warp_img.shape[1]]
        return crop_warp_img, start_y, end_y

    def reverse_crop_warp(self, original_warp, warp_result_cropped, start_y, end_y):
        '''
        reverse image in range of angle to be 0 to 360 degree
        detail: use crop_warp paste into full image or original warp
        :param original_warp:
        :param warp_result_cropped:
        :param start_y:
        :param end_y:
        :return:
        '''
        original_warp[start_y:end_y, 0:original_warp.shape[0]] = warp_result_cropped
        return original_warp

    @staticmethod
    def fit_circle_2d(x, y, w=[]):
        '''
        https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
        fit circle by least square
        MRTHOD: using circle regression to fit the circle by those list of x and list of y
        :param x: all list of x points
        :param y: all list of y points
        :param w:
        :return:
        '''
        x = np.array(x)
        y = np.array(y)
        A = array([x, y, ones(len(x))]).T
        b = x * x.transpose() + y * y.transpose()

        # Modify A,b for weighted least squares
        if len(w) == len(x):
            W = diag(w)
            A = dot(W, A)
            b = dot(W, b)

        # Solve by method of least squares
        c = linalg.lstsq(A, b, rcond=None)[0]

        # Get circle parameters from solution c
        xc = c[0] / 2
        yc = c[1] / 2
        r = sqrt(c[2] + xc ** 2 + yc ** 2)
        return xc, yc, r

    def warp_polar(self, img_bi, circles):
        """
        warp polar : https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga49481ab24fdaa0ffa4d3e63d14c0d5e4
        :param img_bi:
        :param img_ori_result:
        :return:
        """
        warp_img = None
        if circles is not None:
            circle = circles
            # circles = sorted(circles[0],key = lambda s:s[2])
            # circle = circles[-1]
            center = (circle[0], circle[1])
            radius = circle[-1] + int(circle[-1]*(EXTEND_CIRCLE_FACTOR)) #0.03
            dsize = (int(radius), int(2*math.pi*radius))
            warp_img = cv2.warpPolar(img_bi,dsize,center,radius,cv2.WARP_POLAR_LINEAR)
            dst = cv2.warpPolar(warp_img, (img_bi.shape[1], img_bi.shape[0]), center,radius, flags=(cv2.WARP_INVERSE_MAP))
            # cv2.imwrite("output/reverse_original"+str(self.opt.contact_lens.area_upper)+self.name+".jpg",dst)
            self.ori_reverse = dst
            if self.opt.basic.debug == "True":
                imshow_fit("warp_polar__reverse",dst)

        img = img_bi
        return img, warp_img

    def collect_data_area(self, image): # algorithm2 3
        '''
        for the algorithm will divide the edge image to many piece and this function will collect
        area of each piece the store in list
        :param image:
        :return: list of areas
        '''
        stepSize = int(self.opt.contact_lens.filter_size / 2)
        areas = []
        for j, y in enumerate(range(0, image.shape[0], stepSize)):
            im = image[y:y + stepSize, int(image.shape[1] * 0):int(image.shape[1])]
            main_contour,_ = self.select_biggest_contour(im)
            # print(main_contour,"main contour")
            if main_contour != []:
                area = cv2.contourArea(main_contour[0])
                # _,area_in_img = self.select_biggest_contour(im)
                areas.append(area)
            else:
                areas.append(0)

        return areas

    def collect_data_coord(self,image):
        '''

        :param image:
        :return:
        left_coords [x_value of left of contour]
        , right_coords [x_value of right of contour]
        ,center_coords [[x_center, ., .]
                        [y_center, .. . .]]
        '''
        stepSize = int(self.opt.contact_lens.filter_size / 2)
        left_coords = []
        right_coords = []
        x_center_coords = []
        y_center_coords = []
        for j, y in enumerate(range(0, image.shape[0], stepSize)):
            im = image[y:y + stepSize, int(image.shape[1] * 0):int(image.shape[1])]
            main_contour, _ = self.select_biggest_contour(im)
            # if self.opt.basic.debug == "True":
            #     draw = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
            #     cv2.drawContours(draw,main_contour, 0, (0, 0, 255), 1)

            if main_contour != []:
                topleft, botright = contour_to_box(main_contour[0])
                left_coords.append(topleft[0])
                right_coords.append(botright[0])
                x_center_coords.append(int((topleft[0]+botright[0])/2))
                y_center_coords.append(int((topleft[1]+botright[1])/2))

            else:
                left_coords.append(0)
                right_coords.append(0)

        return left_coords, right_coords, [x_center_coords,y_center_coords]

    def collect_data_thickness(self,image):
        '''
        thickness
        or the algorithm will devide the edge to many piece and this function will collect
        thickness of each piece the store in list
        :param image:
        :return: thiickness [t1,t2,t3, . . . ]
        '''

        if self.opt.contact_lens.select_edge_contour == "average":
            pass


        stepSize = int(self.opt.contact_lens.filter_size / 2)

        thicknesses = []
        for j, y in enumerate(range(0, image.shape[0], stepSize)):
            im = image[y:y + stepSize, int(image.shape[1] * 0):int(image.shape[1])]
            main_contour, _ = self.select_biggest_contour(im)
            # if self.opt.basic.debug == "True":
            #     draw = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
            #     cv2.drawContours(draw,main_contour, 0, (0, 0, 255), 1)

            if main_contour != []:
                topleft, botright = contour_to_box(main_contour[0])
                thicknesses.append(botright[0]-topleft[0])
            else:
                thicknesses.append(0)
        return thicknesses

    def collect_data_bitwise_area(self,image):
        # this function will not be used in this time
        '''

        :param image:
        :return:
        '''
        stepSize = int(self.opt.contact_lens.filter_size / 2)
        thicknesses = []
        for j, y in enumerate(range(0, image.shape[0], stepSize)):
            im = image[y:y + stepSize, int(image.shape[1] * 0):int(image.shape[1])]
            main_contour, _ = self.select_biggest_contour(im)
            # if self.opt.basic.debug == "True":
            #     draw = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            #     cv2.drawContours(draw, main_contour, 0, (0, 0, 255), 1)

            if main_contour != []:
                topleft, botright = contour_to_box(main_contour[0])
                thicknesses.append(botright[0] - topleft[0])
            else:
                thicknesses.append(0)
        return thicknesses

    def crop_data_in_region(self,data, iter):
        '''
        collet data around the specify region for example at the 0 degree angle it will count or collect
        -20 degree to 20 degree depends on setting.
        So for this function will collect data from -20 to 20 degrees.
        :param data: list of whole data
        :param iter: index of the region
        :return: list of region data
        '''
        # print(list_areas)
        if iter < self.opt.contact_lens.lower_bound:
            lower = data[len(data) - abs(iter - self.opt.contact_lens.lower_bound):len(data)]
            upper = data[0:iter + self.opt.contact_lens.upper_bound]
            # print(len(lower),"+",len(upper))
            data_in_region = lower + upper

        elif iter > len(data) - self.opt.contact_lens.upper_bound:
            lower = data[iter - self.opt.contact_lens.lower_bound:len(data)]
            upper = data[0: iter +  self.opt.contact_lens.upper_bound - len(data)]
            data_in_region = lower + upper
        else:
            data_in_region = np.array(
                data)[iter - self.opt.contact_lens.lower_bound:iter + self.opt.contact_lens.upper_bound]
        return  data_in_region

    def reverse_warp(self,ori_image,warped_image,circles):
        '''
        TO REVERSE WARP IMAGE
        when using warp polar and want to reverse that warp image.
        :param ori_image:
        :param warped_image:
        :param circles:
        :return:
        '''
        if circles is not None:
            # circles = sorted(circles[0], key=lambda s: s[2])
            circle = circles[-1]
            center = (circle[0], circle[1])
            radius = circle[-1] + int(circle[-1] * (EXTEND_CIRCLE_FACTOR)) # extend circle
            # dsize = (int(radius), int(2 * math.pi * radius))
        reversed_img = cv2.warpPolar(warped_image, (ori_image.shape[1], ori_image.shape[0]), center, radius, flags=(cv2.WARP_INVERSE_MAP))
        return reversed_img

    def area_in_bi_image(self,img):
        '''
        COUNT WHITE AREA IN BINARY IMAGE
        sum of all contour area in binary image
        :param img:
        :return: sum of areas
        '''
        # th = img
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

        contours, __ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            areas += area
        return areas

    def select_biggest_contour(self, img):
        '''
        select biggest contour FROM WHITE PARTS IN BINARY IMAGE
        :param img:
        :return:
        '''
        # th = img
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
        try:

            _, contours, __ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        except:

            contours, __ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = []
        # print(len(contours))
        if contours != []:
            for contour in contours:
                area = cv2.contourArea(contour)
                areas.append(area)
            if areas != []:
                max_area = max(areas)
                for i, area in enumerate(areas):
                    if area == max_area:
                        selected_contour = [contours[i]]
            else:
                selected_contour = contours
        else:
            selected_contour = [array([[[0, 0]], [[0, 0]]], dtype=int32)]

        return selected_contour, areas

    def process_kernel_area(self,image,y):
        '''
        make a judgment of the kernel(small image) whether that is defect or not by area algorithm 1 2 3
        :param image: small image
        :param y:
        :return:
        '''
        img_proc, _, _ = self.reading.read_params(self.kernel_params, image)
        img = img_proc["final"]
        # if self.opt.basic.debug == "True":
        #     cv2.imwrite("output/kernel_th/proc_"+str(y)+self.name+".jpg",img)
        main_contours,_ = self.select_biggest_contour(img)
        if main_contours != []:
            areas = cv2.contourArea(main_contours[0])
        else:
            areas = 0

        # LOWER BOUND self.criterion_kernel[1]
        # UPPER BOUND self.criterion_kernel[0]
        if areas > self.criterion_kernel[1] or areas < self.criterion_kernel[0] :
            mask_red = True
        else:
            mask_red = False
        return mask_red

    def process_kernel_coord(self, image,y):
        '''
        make a judgment of the kernel(small image) whether that is defect or not by coord algorithm4
        :param image: image
        :param y:
        :return:
        '''
        img_proc, _, _ = self.reading.read_params(self.kernel_params, image)
        img = img_proc["final"]
        # img = image
        # _ ,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV)

        # if self.opt.basic.debug == "True":
        #     cv2.imwrite("output/kernel_th/proc_" + str(y) + self.name + ".jpg", img)
        main_contours, _ = self.select_biggest_contour(img)

        if main_contours != []:
            topleft, botright = contour_to_box(main_contours[0])
            left = topleft[0]
            right = botright[0]
        else:
            left = 0
            right = 0
        # LEFT PART
        # LOWER BOUNR  self.criterion_kernel[1] UPPER BOUND self.criterion_kernel[0]
        # LOWER BOUND self.criterion_kernel[3] UPPER BOUND self.criterion_kernel[2]
        if left > self.criterion_kernel[1] or left < self.criterion_kernel[0] or right > self.criterion_kernel[3] or right < self.criterion_kernel[2]:
            mask_red = True
        else:
            mask_red = False
        return mask_red

    def process_kernel_thickness(self,image,y):
        '''
        make a judgment of the kernel(small image) whether that is defect or not by thickness algorithm5
        :param image: small image
        :param y:
        :return:
        '''
        img_proc, _, _ = self.reading.read_params(self.kernel_params, image)
        img = img_proc["final"]
        im = img
        # im = img.copy()
        # checkpoint select biggest contour

        if self.opt.contact_lens.select_edge_contour == "average":
            # th = im
            _, th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV)
            contours, __ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            new_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # print(area)
                if area > 200:
                    # print(area)
                    new_contours.append(contour)
            boxes = self.contours_to_boxes(new_contours)
            index_of_the_contour = self.select_closest_contour(boxes, [self.x_centers_avg, self.y_centers_avg])

            try:
                main_contours = new_contours[index_of_the_contour]
            except:
                main_contours, _ = self.select_biggest_contour(im)
        else:
            main_contours, _ = self.select_biggest_contour(im)

        if main_contours != []:
            topleft, botright = contour_to_box(main_contours[0])
            left = topleft[0]
            right = botright[0]
            thickness = right - left
            # print("thickness: ",thickness)
        else:
            thickness = 999

        # THICKNESS
        # LOWER BOUND self.criterion_kernel[0] UPPER BOUND self.criterion_kernel[1]
        if thickness < self.criterion_kernel[0] or thickness > self.criterion_kernel[1]:
            mask_red = True
        elif thickness == 0 :
            mask_red = True
        else:
            mask_red = False
        return mask_red

    def init_process_algor(self,img,stepSize):
        '''
        TO INITIALIZE THE SELECTED ALGORITHM PROCESS E.G. LIST OF DATA
        ALGORITHM
        1: BAES ON AREA WITH CUTTING PEAK DATA POINT
        2: BASE ON AREA
        3:
        4: BASE ON COORDINATE
        5: BASE ON THICKNESS
        :param img:
        :param stepSize:
        :return:
        '''
        if self.opt.algorithm == 1:
            # all_area = self.area_in_bi_image(img)
            # # average_area = all_area / (int(img.shape[0] / (stepSize - 1)))
            list_areas = self.collect_data_area(img)
            new_area_average = average_in_sd(list_areas)
            self.criterion_kernel = (
            new_area_average * (1 - self.opt.contact_lens.area), new_area_average * (1 + self.opt.contact_lens.area))
            return  list_areas,list_areas
        elif self.opt.algorithm in [2, 3]:
            list_areas = self.collect_data_area(img)
            # plot_data(list_areas, name_save=self.name, save=True)
            # plot_distribution(list_areas, name_save=self.name, save=True)
            return list_areas,list_areas
        elif self.opt.algorithm == 4:
            list_left_data, list_right_data, _ = self.collect_data_coord(img)
            return list_left_data, list_right_data

        elif self.opt.algorithm == 5:

            self.opt.al5.standard_thickness = self.standard_thickness
            if self.opt.contact_lens.select_edge_contour == "average":
                _, _, centers_contour = self.collect_data_coord(img)
                x_centers = np.array(centers_contour[0])
                self.x_centers_avg = np.mean(x_centers)
                self.x_centers_avg = average_in_sd(x_centers,0.5,0.5)
                y_centers = np.array(centers_contour[1])
                self.y_centers_avg = np.mean(y_centers)
                self.y_centers_avg = average_in_sd(y_centers,0.5,0.5)
                # print("AVERAGE CENTER",self.x_centers_avg,self.y_centers_avg)
            print(self.opt.al5.standard_thickness)
            if int(self.opt.al5.standard_thickness) == 0:
                thicknesses = self.collect_data_thickness(img)
                np_thicknesses = np.array(thicknesses)
                self.opt.al5.standard_thickness = np.mean(np_thicknesses)
                # self.opt.al5.standard_thickness = np.mean(np_thicknesses)

            elif int(self.opt.al5.standard_thickness) == 1:
                thicknesses = self.collect_data_thickness(img)
            else:
                pass
            return thicknesses,thicknesses


    def slide_y_window(self, image, window_size):
        '''
        TO MAKE A JUDGEMENT OF POSITIONS OF DEFECT BY CHECKING MANY DIVIDED ELEMENT ON THE EDGE
        :param image:
        :param window_size:
        :return: IMAGE RESULT
        '''
        self.boo_defect = False
        roi = image[0:image.shape[0], int(image.shape[1] * LEFT_BOUND_WARP_IMG):int(image.shape[1])]
        self.roi = roi.copy()
        # if self.opt.basic.debug == "True":
        #     imshow_fit("slide_y_window__edge", roi)
        # img0 = image.copy()
        stepSize = int(window_size / 2)
        img_draw = image.copy()
        img_proc, _, _ = self.reading.read_params(self.kernel_params, roi)
        img = img_proc["final"]
        list_Areas_Leftcoord_Thickness ,List_Right = self.init_process_algor(img,stepSize)
        for j, y in enumerate(range(0, image.shape[0], stepSize)):
            im = image[y:y + stepSize, int(image.shape[1] * LEFT_BOUND_WARP_IMG):int(image.shape[1])]
            # if self.opt.basic.debug == "True":
            # cv2.imwrite("./output/kernel_ori/ori_" + self.name + "_" + str(y) + ".jpg", im)

            # im = image[y:y + stepSize, int(image.shape[1] * 0):int(image.shape[1])]
            if self.opt.algorithm in [2, 1]:
                list_areas = list_Areas_Leftcoord_Thickness
                temporary = np.array(
                    list_areas[j - self.opt.contact_lens.lower_bound:j + self.opt.contact_lens.upper_bound])
                average_area = np.mean(temporary)

                self.criterion_kernel = (average_area * (1 - self.opt.contact_lens.area_lower),
                                         average_area * (1 + self.opt.contact_lens.area_upper))
                mask = self.process_kernel_area(im, y)

            elif self.opt.algorithm == 3:
                list_areas = list_Areas_Leftcoord_Thickness
                temporary = self.crop_data_in_region(list_areas, j)
                # average_area = average_in_sd(temporary, sd_lower=1, sd_upper=1)
                average_area = average_in_sd(temporary, sd_lower=10, sd_upper=10)
                self.criterion_kernel = (average_area * (1 - self.opt.contact_lens.area_lower),
                                         average_area * (1 + self.opt.contact_lens.area_upper))
                mask = self.process_kernel_area(im, y)

            elif self.opt.algorithm == 4:
                list_left_data, list_right_data = list_Areas_Leftcoord_Thickness,List_Right
                # print("list_left_data", list_left_data,"list_right_data",list_right_data)
                left_temporary = self.crop_data_in_region(list_left_data, j)
                right_temporary = self.crop_data_in_region(list_right_data, j)
                left_average_area = average_in_sd(left_temporary, sd_lower=0.5, sd_upper=0.5) + 1  # +1 correction
                right_average_area = average_in_sd(right_temporary, sd_lower=0.5, sd_upper=0.5) - 1  # -1 correction
                self.criterion_kernel = (left_average_area - self.opt.contact_lens.al4.delta_pix,
                                         left_average_area + self.opt.contact_lens.al4.delta_pix,
                                         right_average_area - self.opt.contact_lens.al4.delta_pix,
                                         right_average_area + self.opt.contact_lens.al4.delta_pix)
                mask = self.process_kernel_coord(im, y)

            elif self.opt.algorithm == 5:
                thicknesses = list_Areas_Leftcoord_Thickness
                if int(self.opt.al5.standard_thickness) in [0, 1]:
                    thickness_temporary = self.crop_data_in_region(thicknesses, j)
                    thicknesses_average = average_in_sd(thickness_temporary,sd_lower=0, sd_upper=0)
                    # thicknesses_average = average_in_sd(thickness_temporary,sd_lower=0.5, sd_upper=0.5)
                    # print("thicknesses_average",thicknesses_average.shape)
                    if thicknesses_average.shape == ():
                        thicknesses_average = thickness_temporary
                    self.opt.al5.standard_thickness = np.mean(np.array(thicknesses_average))
                self.criterion_kernel = (self.opt.al5.standard_thickness - self.opt.al5.shrink,
                                         self.opt.al5.standard_thickness + self.opt.al5.extended)
                # print(self.criterion_kernel)
                mask = self.process_kernel_thickness(im, y)

            if mask:
                cv2.rectangle(img_draw, (int(image.shape[1] * 0.95), y), (int(image.shape[1]), y + stepSize),
                              (0, 0, 255), 20)
                self.boo_defect = True
        return img_draw

    def select_average_position_contour(self):
        pass

    def set_params_files(self):
        '''
        TO READ SETTING JSON FILES TO BE PARAMETERS FOR THIS AOI
        :return:
        '''
        ind_param = 0
        try:
            with Path(self.opt["basic"]["path_params" + str(ind_param)]).open("r") as f:
                self.log.show("read params.json", "DEBUG")
                self.params = json.load(f)
                if self.opt.basic.debug == "True":
                    print("params",self.params)
        except:
            self.log.show(self.opt["basic"][
                              "path_params" + str(ind_param)] + " doesn't exist. Please, select the file again!",
                          "ERROR")
        try:
            with Path(self.opt["basic"]["path_kernel" + str(ind_param)]).open("r") as f:
                self.log.show("read params.json", "DEBUG")
                self.kernel_params = json.load(f)
                if self.opt.basic.debug == "True":
                    print("params",self.kernel_params)
        except:
            self.log.show(self.opt["basic"][
                              "path_kernel" + str(ind_param)] + " doesn't exist. Please, select the file again!",
                          "ERROR")
        self.standard_thickness = self.opt.al5.standard_thickness


    def fov1_process(self,name,source):
        '''

        :param name:
        :param source:
        :return:
        '''
        key = ord("a")
        self.name = name[0:-4]

        if not os.path.isdir("output/al"+str(self.opt.algorithm) +"/" ):
            os.mkdir("output/al"+str(self.opt.algorithm) +"/" )
        if not os.path.isdir("output/al"+str(self.opt.algorithm) +"/" + self.name + "/"):
            os.mkdir("output/al"+str(self.opt.algorithm) +"/" + self.name + "/")

        while key != ord("q"):
            frame1 = cv2.imread(source + "/" + name)
            self.original_image_to_draw = deepcopy(frame1)
            if self.opt.basic.debug == "True":
                imshow_fit("fov1_process__frame1", frame1)
            self.set_params_files()
            time0 = time.time()
            state = self.fov1_exceute(frame1)
            imshow_fit("RESULT",state)
            time1 = time.time()
            print("Exceution time : ",time1-time0)
            key = cv2.waitKey(self.opt.basic.waitkey)
            if self.opt.basic.waitkey == 1:
                key = ord("q")

    def fov1_exceute(self,frame_ori):
        try:
            print(self.name)
        except:
            self.name = "testing"
        yolo_result = []
        frame = frame_ori.copy()
        # image processing
        frame0 = frame.copy()

        # image processing by reading json
        img_proc, circle, _ = self.reading.read_params(self.params, frame)
        img_params = img_proc["final"]  # for detecting circle
        _, img_params = cv2.threshold(img_params, 0, 255, cv2.THRESH_BINARY_INV)  # inverse th
        self.img_params = img_params.copy()
        try:
            _,contours, _ = cv2.findContours(img_params, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        except:

            contours, _ = cv2.findContours(img_params, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        filter_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.opt.threshold.area_contour_circle[0] and area < self.opt.threshold.area_contour_circle[1]:
                if self.opt.basic.debug == "True":
                    cv2.drawContours(frame, [contour], -1, (255, 255, 255), 3)
                    print("area in circle detection threshold",area)
                    imshow_fit("fov1_execute__img_params", img_params)
                    imshow_fit("fov1_execute__test_contour_area", frame)
                filter_contours.append(contour)

        self.warp_polar_result = []
        self.circle = (0,0,0)
        if filter_contours != []:
            get_boxes = self.contours_to_boxes(filter_contours)
            # select edge contour
            circles = []
            for index in self.select_direction_contour(filter_contours, get_boxes, direction="low"):
                lowest_contour = filter_contours[index]
                xs, ys = self.__get_x_y_from_contour(lowest_contour)
                cen_x, cen_y, r = self.fit_circle_2d(xs, ys)

                if self.opt.basic.debug == "True":
                    cv2.circle(frame0, (int(cen_x + int(frame_ori.shape[1] * CROP_CENTERIMAGE)), int(cen_y)), int(r),
                               (255, 0, 0), 10)
                    imshow_fit("fov2_execute__CIRCLES", frame0)
                    print("radius : ", r)
                    print("center :", cen_x, ",", cen_y)
                if r < self.opt.threshold.range_radius[1] and r > self.opt.threshold.range_radius[0]:
                    circles.append((cen_x, cen_y, r))
                elif r < 1190 and r > 1100:
                    circles.append((cen_x, cen_y, 1210))

                # if r < 1300 and r > 1190:
                # if r < 1490 and r > 1300:
                #     circles.append((cen_x, cen_y, r))
                #     print("radius : ", r)
                # elif r < 1190 and r > 1100:
                #     circles.append((cen_x, cen_y, 1210))

            # print("circle ", len(circles))
            circles = sorted(circles, key=lambda s:s[2])

            if len(circles) <= 10 and len(circles)>0:
                cen_x, cen_y, r = circles[-1]
                # print("center",cen_x,",", cen_y,"\nradius",r)
                # cv2.circle(frame1,( int(cen_x+int(frame1.shape[1]*CROP_CENTERIMAGE)),int(cen_y)),int(r),(255,0,0),10)

                # ori_cen_x,ori_cen_y, ori_r = cen_x*self.opt.basic.resize_factor,cen_y*self.opt.basic.resize_factor, r*self.opt.basic.resize_factor
                self.circle = circles
                img, warp = self.warp_polar(frame_ori, [cen_x + int(frame_ori.shape[1] * CROP_CENTERIMAGE), cen_y, r])
                warp_save = warp.copy()
                crop_warp, strat_y, end_y = self.crop_warp(warp, 0, 360)
                crop_warp_save = crop_warp.copy()

                reversed_crop_warp = self.reverse_crop_warp(warp, crop_warp, strat_y, end_y)

                self.ori_reverse = self.reverse_warp(frame_ori, reversed_crop_warp,
                                                     [[cen_x + int(frame_ori.shape[1] * CROP_CENTERIMAGE), cen_y, r]])
                ori_rev_copy = self.ori_reverse.copy()


                warp_result = self.slide_y_window(crop_warp, self.opt.contact_lens.filter_size)
                # warp_result = self.reverse_crop_warp(warp, warp_result, strat_y, end_y)
                self.warp_polar_result = warp_result
                reversed_image = self.reverse_warp(frame_ori, warp_result,
                                                   [[cen_x + int(frame_ori.shape[1] * CROP_CENTERIMAGE), cen_y, r]])
                if self.opt.algorithm in [1, 2, 3]:
                    box_proc = True
                else:
                    box_proc = False

                if self.opt.basic.yolo_result == "True":
                    yolo_result = draw_box(reversed_image, self.ori_reverse, self.params_box, box_proc)
                    yolo_result_ori = draw_box(reversed_image,self.original_image_to_draw,  self.params_box, box_proc)
                    cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                        self.opt.contact_lens.area_upper) + self.name + "box_result_ori.jpg", self.original_image_to_draw)

                else:
                    yolo_result = reversed_image

                if self.opt.basic.debug == "True":
                    imshow_fit("contour", reversed_image)
                    imshow_fit("fov1_execute__crop_warp", self.ori_reverse)
                    # cv2.imwrite("drawcircle.jpg", frame1)
                    cv2.imwrite("testqqqq.jpg", crop_warp)


                    if self.opt.algorithm in [1, 2, 3]:
                        print("SAVE to "+ "output/al" + str(self.opt.algorithm) + "/")
                        # save warp
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                            self.opt.contact_lens.area_upper) + self.name + "img_params.jpg", self.img_params)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "warp_result.jpg", warp_result)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "warp.jpg", warp)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "ROI.jpg", self.roi)
                        # save ori_reverse
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + "original" + str(
                            self.opt.contact_lens.area_upper) + self.name + ".jpg", ori_rev_copy)
                        # save result
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + "result" + str(
                            self.opt.contact_lens.area_upper) + "/" + self.name + ".jpg", reversed_image)
                        cv2.imwrite(
                            "output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                                self.opt.contact_lens.area_upper) + "_" + str(
                                self.opt.contact_lens.area_lower) + "_" + self.name + ".jpg",
                            yolo_result)
                    elif self.opt.algorithm in [4]:
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                            self.opt.contact_lens.area_upper) + self.name + "img_params.jpg", self.img_params)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.al4.delta_pix) + self.name + "warp.jpg", warp_result)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "warp.jpg", warp)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "ROI.jpg", self.roi)
                        # save ori_reverse
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + "original" + str(
                            self.opt.contact_lens.al4.delta_pix) + self.name + ".jpg", ori_rev_copy)
                        # save result
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+"result" + str(
                            self.opt.contact_lens.al4.delta_pix) + self.name + ".jpg", reversed_image)
                        #
                        cv2.imwrite(
                            "output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                                self.opt.contact_lens.al4.delta_pix) + "_" + self.name + ".jpg",
                            yolo_result)
                    elif self.opt.algorithm in [5] :
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "img_params.jpg", self.img_params)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                            self.opt.al5.extended) + "_" + str(
                            self.opt.al5.shrink) + self.name + "warp.jpg", warp_result)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "warp.jpg", warp)
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/"+ str(
                            self.opt.contact_lens.area_upper) + self.name + "ROI.jpg", self.roi)
                        # save ori_reverse
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + "original" + str(
                            self.opt.al5.extended) + "_" + str(self.opt.al5.shrink)
                                    + self.name + ".jpg", ori_rev_copy)
                        # save result
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + "result" +
                                    str(self.opt.al5.extended) + "_" + str(self.opt.al5.shrink) + self.name + ".jpg",
                                    reversed_image)
                        #
                        cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                            self.opt.al5.extended) + "_" + str(self.opt.al5.shrink) + "_" + self.name + ".jpg", yolo_result)
                        pass
            else:
                print("defect!!! or cannot detect contact lens")

                yolo_result = frame0
                if self.opt.basic.debug == "True":
                    cv2.putText(frame0, "Defect", (0, frame_ori.shape[0]), 0, 20, [225, 0, 0], thickness=20,
                                lineType=cv2.LINE_AA)
                    imshow_fit("DEFECT", frame0)
                    cv2.imwrite("output/al" + str(self.opt.algorithm) + "/" + self.name + "/" + str(
                        self.opt.al5.extended) + "_" + str(
                        self.opt.al5.shrink) + "_" + self.name + "defect" + ".jpg", frame0)
                    # continue
                # cv2.circle(frame_ori,( int(cen_x+int(frame_ori.shape[1]*CROP_CENTERIMAGE)),int(cen_y)),int(r),(255,0,0),10)
                    cv2.imwrite("drawcircle.jpg", frame_ori)
            # print("breaks")
            # break


        else:
            self.log.show("cannot detect circle, please check light source ", "WARNING")

        return yolo_result


    def fov1_specified_result(self):
        '''

        :return: linear image , circle value (x,y,r)
        '''
        return self.warp_polar_result, self.circle

    def main(self):
        """
        Function Name: main

        Description: Connect all part together

        Argument:
            params [[type]] -> [description]
            rect_params [[type]] -> [description]

        Parameters:

        Return:

        Edited by: [date] [author name]
        """

        source = self.opt.basic.source
        webcam = source.isnumeric()

        if webcam:
            self.log.show("<=== initialize webcam "+source+" ===>", "INFO")
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.opt.basic.web_cam.width)  # add in opt
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.opt.basic.web_cam.height)  # add in opt
            cap.set(cv2.CAP_PROP_FPS, self.opt.basic.web_cam.fps)  # add in opt

        elif source == "pylon":
            self.log.show("<=== initialize pylon camera ===>", "INFO")
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            converter = pylon.ImageFormatConverter()

            # ========== Grabing Continusely (video) with minimal delay ==========
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # ========== converting to opencv bgr format ==========
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            rng.seed(12345)
        else:
            self.log.show("read files in " + self.opt.basic.source, "INFO")
            im_name = listDir(source)
            for name in im_name:
                # key = ord("a")
                print(name)
                if self.opt.test.loop_test == "True":
                    for i in range(self.opt.test.n_steps):
                        try:
                            with Path("config/main.json").open("r") as f:
                                self.opt = json.load(f)
                                # print(self.opt)
                                self.opt = EasyDict(self.opt)
                        except:
                            self.log.show(
                                "config/main.json" + " doesn't exist. Please, select the file again!",
                                "ERROR")
                        if self.opt.algorithm in [1,2,3]:
                            self.opt.contact_lens.area_upper = INITIAL_THRESHOLD+self.opt.test.step*i
                            self.opt.contact_lens.area_lower = INITIAL_THRESHOLD+self.opt.test.step*i
                        elif self.opt.algorithm in [4]:
                            self.opt.contact_lens.al4.delta_pix = INITIAL_PIXEL+self.opt.test.pix_step*i
                        elif self.opt.algorithm in [5]:
                            self.opt.al5.extended = self.opt.al5.extended + self.opt.test.pix_step*i
                            self.opt.al5.shrink = self.opt.al5.shrink + self.opt.test.pix_step*i
                        t0 = time.time()
                        self.fov1_process(name,source)
                        print("processing time:",time.time()-t0)
                else:
                    self.fov1_process(name,source)



if __name__ == "__main__":

    fov1 = Fov1()

    for i in range(2,5):
        print(i)
        fov1.opt["algorithm"] = i
        fov1.main()