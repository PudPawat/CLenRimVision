from .improc_save import Imageprocessing
import cv2

class read_save(object):
    def __init__(self):
        """
        Function Name: __init__
        
        Description: read_save this class has functions for reading saved information to repeatly use in main. 
        
        Argument:
        
        Parameters:
        
        Return:
        
        Edited by: [12-07-2020] [Pawat]
        """        
        self.imgproc = Imageprocessing()


    def read_params(self, params, frame,print=False):
        """
        Function Name: read_params
        
        Description: read all image processing from json file eg. threshold, HSV inrange
        and put those parameters to process in Imageorocessing() 
        
        Argument:
            params [dict] -> [all parameters]
            frame [array] -> [image for processing]
        
        Parameters:
        
        Return:
            frame [array] -> [image after process]
        
        Edited by: [12-07-2020] [Pawat]
        """
        frame_proc = {}
        circle = {}
        line = {}
        for key in params.keys():
            if print:
                print(key)
            circle = []
            line = []
            # frame_result1 = frame.copy()
            # frame = cv2.resize(frame_result1, (int(frame_result1.shape[1] / self.opt.basic.resize_factor),
            #                                   int(frame_result1.shape[0] / self.opt.basic.resize_factor)))
            # frame_result = frame.copy()
            if key == "HSV":
                # frame_HSV, params['HSV'] = imgproc.HSV_range(frame, params[key])
                frame, params['HSV'] = self.imgproc.HSV_range(frame, params[key])
                frame_proc["HSV"] = frame

            elif key == "erode":
                # frame_erode, params['erode'] = imgproc.erode(frame, params[key])
                frame, params['erode'] = self.imgproc.erode(frame, params[key])
                frame_proc["erode"] = frame

            elif key == "dilate":
                # frame_dialte, params['dilate'] = imgproc.dilate(frame, params[key])
                frame, params['dilate'] = self.imgproc.dilate(frame, params[key])
                frame_proc["dilate"] = frame

            elif key == "thresh":
                # frame_binary, params['thresh'] = imgproc.threshold(frame, params[key])
                frame, params['thresh'] = self.imgproc.threshold(frame, params[key])
                frame_proc["thresh"] = frame

            elif key == "sharp":
                # frame_sharp, params['sharp'] = imgproc.sharpen(frame, params[key])
                frame, params['sharp'] = self.imgproc.sharpen(frame, params[key])
                frame_proc["sharp"] = frame

            elif key == "blur":
                # frame_blur, params['blur'] = imgproc.blur(frame, params[key])
                frame, params['blur'] = self.imgproc.blur(frame, params[key])
                frame_proc["blur"] = frame

            elif key == "line":
                # frame_line, lines, params['line'] = imgproc.line_detection(frame, frame0, params[key])
                if len(frame.shape) == 2:
                    frame0 = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame, lines, params['line'] = self.imgproc.line_detection(frame, frame0, params[key])
                frame_proc["line"] = frame

            elif key == "canny":
                # frame_canny, params['canny'] = imgproc.canny(frame, params[key], show=True)
                frame, params['canny'] = self.imgproc.canny(frame, params[key], show=False)
                frame_proc["canny"] = frame

            elif key == "circle":
                # frame_circle, circle, params['circle'] = imgproc.circle_detection(frame, frame0, params[key], show=False)
                if len(frame.shape) == 2:
                    frame0 = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame, circle, params['circle'] = self.imgproc.circle_detection(frame, frame0, params[key], show=False)
                frame_proc["circle"] = frame
        frame_proc["final"] = frame

        return frame_proc, circle, line

    def read_rectangle(self,config,img):
        """
        Function Name: read_rectangle
        
        Description: read rectangles those are selected in setting_rects.py and save in json file format
        in the input format includes "main" which means a rectangle of head of wire and "sub" which means \
        rectangles in the main rectangle(head of the wire). 
        This fuction for showing the region in real-time inspection to check weather or not those inspection wires
        are in the right positions.
        
        Argument:
            config [dict] -> [all main rectangles and sub rectangles]
            img [array] -> [image to write rectangles]
        
        Parameters:
        
        Return:
            img[array] -> [written image]
        
        Edited by: [12-07-2020] [author name]
        """        
        for i in range(len(config['main'])):
            main = config['main'][str(i)]
            # imCrop = im[int(main[1]):int(main[1] + main[3]), int(main[0]):int(main[0] + main[2])]
            y0 = int(main[1])
            y1 = int(main[1] + main[3])
            x0 = int(main[0])
            x1 = int(main[0] + main[2])
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 3)

            for j in range(len(config['sub'][str(i)])):
                sub = config['sub'][str(i)][str(j)]
                sub_y0 = int(sub[1])
                sub_y1 = int(sub[1] + sub[3])
                sub_x0 = int(sub[0])
                sub_x1 = int(sub[0] + sub[2])
                cv2.rectangle(img, (x0 + sub_x0, y0 + sub_y0), (x0 + sub_x1, y0 + sub_y1), (0, 0, 255), 3)
        return img