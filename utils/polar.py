import cv2
import math


def reverse_warp( ori_image, warped_image, circles):
    if circles is not None:
        circles = sorted(circles[0], key=lambda s: s[2])
        circle = circles[-1]
        center = (circle[0], circle[1])
        radius = circle[-1] + int(circle[-1] * (0.03))
        dsize = (int(radius), int(2 * math.pi * radius))
    reversed_img = cv2.warpPolar(warped_image, (ori_image.shape[1], ori_image.shape[0]), center, radius,
                                 flags=(cv2.WARP_INVERSE_MAP))
    return reversed_img

def warppolar(img_bi, circles):
    """
    warp polar : https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga49481ab24fdaa0ffa4d3e63d14c0d5e4
    :param img_bi:
    :param img_ori_result:
    :return:
    """
    warp_img = None
    if circles is not None:
        circles = sorted(circles[0],key = lambda s:s[2])
        circle = circles[-1]
        center = (circle[0], circle[1])
        radius = circle[-1] + int(circle[-1]*(0.03))
        dsize = (int(radius), int(2*math.pi*radius))
        warp_img = cv2.warpPolar(img_bi,dsize,center,radius,cv2.WARP_POLAR_LINEAR)
    img = img_bi
    return img, warp_img