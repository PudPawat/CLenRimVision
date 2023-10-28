import cv2 as cv
import numpy as np
import math
from lib.trackbar import *


class Imageprocessing(object):

    def __init__(self):
        pass

    def threshold(self, img, params, show = False):
        """
        Function Name: threshold
        
        Description: setting threshold value
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [thresholded image]
            params[tuple] -> (th_val)
        
        Edited by: [12-4-2020] [Pawat]
        """        
        th_val = params
        if th_val == 0 :
            flag = cv.THRESH_BINARY+cv.THRESH_OTSU

        else:
            flag = cv.THRESH_BINARY

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, th = cv.threshold(img,th_val, 255, flag)
        if show == True:
            cv.imshow("window_thresh", th)

        return th, (th_val)

    def canny(self,img,params, show = False):
        """
        Function Name: canny
        
        Description: Canny edge detection. there is two params X ,Y
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [edge image]
            params[tuple] -> (Y_val, X_val)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        Y_val, X_val = params

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny = cv.Canny(img, Y_val, X_val)
        if show == True:
            cv.imshow("window_canny", canny)
        return canny, (Y_val, X_val)

    def canny_1(self,img,params, show = False):
        """
        Function Name: canny_1
        
        Description: another for parallel processing. there is two params X ,Y
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [edge image]
            params[tuple] -> (Y_val, X_val)
        
        Edited by: [12-04-2020] [Pawat]
        """ 
        Y_val, X_val = params

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny = cv.Canny(img, Y_val, X_val)
        if show == True:
            cv.imshow("var_canny_1", canny)
        return canny, (Y_val, X_val)

    def blur(self,img,params, show = False):
        """
        Function Name: blur
        
        Description: Bluring image by setting filter size
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [blured image]
            params[tuple] -> (filter_size)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        filter_size = params
        if filter_size < 1 :
            filter_size = 1

        blur = cv.blur(img, (int(filter_size), int(filter_size)))

        if show == True:
            cv.imshow("window_blur", blur)

        return blur,(filter_size)

    def HSV_range(self,img,params,show = False, mode = "HSV"):
        """
        Function Name: HSV_range
        
        Description: HSV thresholding by setting lower bound and upper bound
        of Hue, Satuation, Value channel
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [thresholded by HSV image]
            params[tuple] -> (low_H, low_S, low_V, high_H, high_S, high_V)
        
        Edited by: [12-04-2020] [Pawat]
        """ 
        low_H, low_S, low_V, high_H, high_S, high_V = params


        if mode == "HSV":
            frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # frame_HSV = cv.cvtColor(frame_HSV, cv.COLOR_HSV2BGR)

            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        elif mode == "HLS":
            frame_HLS = cv.cvtColor(img, cv.COLOR_BGR2HLS)
            frame_threshold = cv.inRange(frame_HLS, (low_H, low_V, low_S), (high_H, high_V, high_S))

        if show == True:
            cv.imshow("window_HSV", frame_threshold)

        return  frame_threshold, [low_H, low_S, low_V, high_H, high_S, high_V]

    def HSV_range_1(self,img ,params ,show = False, mode = "HSV"):
        """
        Function Name: HSV_range
        
        Description: HSV thresholding by setting lower bound and upper bound
        of Hue, Satuation, Value channel
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [thresholded by HSV image]
            params[tuple] -> (low_H, low_S, low_V, high_H, high_S, high_V)
        
        Edited by: [12-04-2020] [Pawat]
        """
        low_H, low_S, low_V, high_H, high_S, high_V = params

        if mode == "HSV":
            frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # frame_HSV = cv.cvtColor(frame_HSV, cv.COLOR_HSV2BGR)

            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        elif mode == "HLS":
            frame_HLS = cv.cvtColor(img, cv.COLOR_BGR2HLS)
            frame_threshold = cv.inRange(frame_HLS, (low_H, low_V, low_S), (high_H, high_V, high_S))

        if show == True:
            cv.imshow("window_HSV1", frame_threshold)

        return  frame_threshold, [low_H, low_S, low_V, high_H, high_S, high_V]


    def HSV_adjustment(self, img, factor_H, factor_S, factor_V):

        frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # checkpoint to continue
        # purpose of this function is adjust like lighrroom

    def sharpen(self,img,params, show = False):
        """
        Function Name: sharpen
        
        Description: Shapen process by factor multiply by filter
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [sharpen image]
            params[tuple] -> (factor)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        factor = params
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        kernel = (factor/10) * kernel
        img = cv.filter2D(img, -1, kernel)
        # if show == True:
        #     cv.imshow(self.var_sharpen.window_sharp_name, img)
        # checkpoint to continue
        return img, (factor)


    def line_detection(self, img, draw_img,params, show = False):
        """
        Function Name: line_detection
        
        Description: Line detection
        with the following arguments:
        dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
        lines: A vector that will store the parameters (r,θ) of the detected lines
        rho : The resolution of the parameter r in pixels. We use 1 pixel. ( 1 to 10 )
        theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180) ( 30 - 180 )
        threshold: The minimum number of intersections to "*detect*" a line
        srn and stn: Default parameters to zero. Check OpenCV reference for more info.
        
        Argument:
            img [array] -> [image for detection]
            draw_img [array] -> [image for drawing]
            params [tuple] -> [parameters]
        
        Parameters:
        
        Return:
            image [array] -> [drown image]
            line [list] -> [list of detected line]
            params[tuple] -> (rho1, theta2, threshold3, none4, srn5, stn6)
        
        Edited by: [12-04-2020] [Pawat]
        """
        # copy_img = img.copy()
        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # copy_img = img.copy()
        rho1, theta2, threshold3, none4, srn5, stn6 = self.var_line_det.return_var()
        if rho1 == 0:
            rho1 = 1
        if theta2 == 0:
            theta2 = 1
        lines = cv.HoughLines(img, rho1, np.pi / theta2, threshold3, None, srn5, stn6)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(draw_img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        if show == True:
            cv.imshow("window_line_detection", draw_img)

        return draw_img,lines , (rho1, theta2, threshold3, none4, srn5, stn6)

    def circle_detection(self, img,draw_img,params ,show= False):
        """
        Function Name: circle_detection
        
        Description: Circle detection by opencv
        
        Argument:
            img [array] -> [image for detection]
            draw_img [array] -> [image for drawing]
            params [tuple] -> [parameters]
        
        Parameters:
        
        Return:
            img[array] -> [drawn image]
            circle[list] -> list of circle
            params[tuple] -> (param1,param2, min, max)
        
        Edited by: [12-04-2020] [Pawat]
        """  

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rows =  img.shape[0]

        param1,param2, min,max = params

        if param1 == 0:
            param1 = 1
        if param2 == 0:
            param2 = 1
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, rows / 5,
                                   param1=param1, param2=param2,
                                   minRadius=min, maxRadius=max)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(draw_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(draw_img, center, radius, (255, 0, 255), 3)


        if show == True:
            cv.namedWindow("window_circle",cv.WINDOW_NORMAL)
            cv.imshow("window_circle", draw_img)

        return img, circles, (param1,param2, min, max)

    def circle_detection_1(self, img,draw_img,params, show= False):
        """
        Function Name: circle_detection_1
        
        Description: Circle detection by opencv
        
        Argument:
            img [array] -> [image for detection]
            draw_img [array] -> [image for drawing]
            params [tuple] -> [parameters]
        
        Parameters:
        
        Return:
            img[array] -> [drawn image]
            circle[list] -> list of circle
            params[tuple] -> (param1,param2, min, max)
        
        Edited by: [12-04-2020] [Pawat]
        """

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rows =  img.shape[0]

        param1,param2, min, max = params

        if param1 == 0:
            param1 = 1
        if param2 == 0:
            param2 = 1
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, rows / 5,
                                   param1=param1, param2=param2,
                                   minRadius=min, maxRadius=max)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(draw_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(draw_img, center, radius, (255, 0, 255), 3)


        if show == True:
            cv.imshow("window_circle1", draw_img)

        return img, circles, (param1,param2, min, max)

    def dilate(self, img,params, show = False):
        """
        Function Name: dilate
        
        Description: dilation processing :making white parts bigger follow kernel shape and size
         type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [dilated image]
            params[tuple] -> (kernel_size, type_kernel)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        # print("Note : \n type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE")
        kernel_size, type_kernel = params

        if type_kernel == 1:
            type_kernel = cv.MORPH_RECT #ok
        elif type_kernel == 2:
            type_kernel = cv.MORPH_OPEN #ok
        elif type_kernel == 3:
            type_kernel = cv.MORPH_CROSS #ok
        elif type_kernel == 4:
            type_kernel = cv.MORPH_DILATE #ok
        elif type_kernel == 5:
            type_kernel = cv.MORPH_ERODE #ok
        elif type_kernel == 6:
            type_kernel = cv.MORPH_ELLIPSE #ok
        else:
            type_kernel = cv.MORPH_ELLIPSE

        if kernel_size == 0:
            kernel_size = 1
        kernel = cv.getStructuringElement(type_kernel, (kernel_size, kernel_size))

        dialate = cv.dilate(img, kernel, iterations=1)

        if show == True:
            cv.imshow("window_dilate_det", dialate)

        return dialate, (kernel_size, type_kernel)


    def erode(self, img,params, show = False):
        """
        Function Name: erode
        
        Description: making white parts smaller follow kernel shape and size
        type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            image[array] -> [eroded image]
            params[array] -> [kernel_size, type_kernel]
        
        Edited by: [12-04-2020] [Pawat]
        """        
        # print("Note : \n type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE")
        kernel_size, type_kernel = params
        # "ty:1REC,2GRA,3Cro,4DIA,5SQR,6STA,7ELIP"
        if type_kernel == 1:
            type_kernel = cv.MORPH_RECT  # ok
        elif type_kernel == 2:
            type_kernel = cv.MORPH_OPEN  # ok
        elif type_kernel == 3:
            type_kernel = cv.MORPH_CROSS  # ok
        elif type_kernel == 4:
            type_kernel = cv.MORPH_DILATE  # ok
        elif type_kernel == 5:
            type_kernel = cv.MORPH_ERODE  # ok
        elif type_kernel == 6:
            type_kernel = cv.MORPH_ELLIPSE  # ok
        else:
            type_kernel = cv.MORPH_ELLIPSE

        if kernel_size == 0 :
            kernel_size = 1
        kernel = cv.getStructuringElement(type_kernel, (kernel_size, kernel_size))
        # kernel = cv.getStructuringElement(type_kernel, (2 * kernel_size + 1, 2 * kernel_size + 1),
        #                                    (kernel_size, kernel_size))
        erode = cv.erode(img, kernel)
        if show == True:
            cv.imshow("window_erode", erode)

        return erode, (kernel_size, type_kernel)

