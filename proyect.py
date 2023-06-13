import numpy as np
import cv2
import math
import random
import time
from lines import lanelines


histo_treshold=40
leftpeak=129
rightpeak=440
leftx_base=0
rightx_base=0
left_flag=0
right_flag=0
car_out=0
anchor_Vehicle = 155#155
laneline_obj = None

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def abs_sobel_thresh(img, orient='x',sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    #binary_output = ((gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])) =1
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def unwarp(img, src, dst):
    h,w = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv



def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    global car_out,left_flag, right_flag,leftpeak,rightpeak
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)
    #print(ploty)
    #left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    #right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
    left_fitx = np.ones_like(ploty) * l_fit[0]
    right_fitx = np.ones_like(ploty) * r_fit[0]
    #print(left_fitx)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    if car_out == 0:
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, ))
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (0,0, 255))
    if left_flag :
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    if right_flag :
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
    
    cv2.line(color_warp, (rightpeak, 0), (rightpeak, h), (255,255,100), thickness=2)
    cv2.line(color_warp, (leftpeak, 0), (leftpeak, h), (255,255,100), thickness=2)
    cv2.imshow("image wrap",color_warp)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


def process_image(image):
    global leftx_base, rightx_base, left_flag, right_flag, car_out,leftpeak,rightpeak,anchor_Vehicle,laneline_obj
    imshape = image.shape
    ksize = 7 
    poly_int=[(0, round(imshape[0]*0.9)),
            (round(imshape[1]*0.36), round(imshape[0]*0.5)),
            (round(imshape[1]*0.62), round(imshape[0]*0.5)),
            (imshape[1], round(imshape[0]*0.9))]
    poly_out=[(0+200,imshape[0]),
              (0+200, 0),
              (imshape[1]-200, 0),
              (imshape[1]-200, imshape[0])]
    regionFindingLane = np.array(poly_int, dtype=np.int32)
    src_region = np.array(poly_int, dtype=np.float32)
    dst_region =  np.array(poly_out, dtype=np.float32)
    roi_img = np.copy(image)
    roi_mask = np.zeros((imshape[0],imshape[1]),np.uint8)
    cv2.fillPoly(roi_mask, pts=[np.rint(poly_int).astype(int)], color=255)
    roi_img = cv2.bitwise_and(roi_img,roi_img,mask=roi_mask)
    cv2.imshow("roi",roi_img)


    img_unwarp, M, Minv = unwarp(image, src_region, dst_region)
    cv2.imshow("unwrap", img_unwarp)

    gradx = abs_sobel_thresh(img_unwarp, orient='x', sobel_kernel=ksize, thresh=(50, 255))# 0 a 255
    #grady = abs_sobel_thresh(img_unwarp, orient='y', sobel_kernel=ksize, thresh=(50, 100))# 0 a 255
    #mag_binary = mag_thresh(img_unwarp, sobel_kernel=ksize, mag_thresh=(50, 255))# 0 a 255
    #dir_binary = dir_threshold(img_unwarp, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))# 0 a pi/2
    s_binary = hls_select(img_unwarp, thresh=(30, 255))
    color_binary = np.dstack(( np.zeros_like(gradx), gradx, s_binary)) * 255
    cv2.polylines(image, [regionFindingLane], True, (255, 255, 0), 1)
    
    #cv2.imshow("hls", s_binary*255)
    cv2.imshow("input video video", image)
    mask = np.zeros((gradx.shape[0],gradx.shape[1]), np.uint8)
    mask[(img_unwarp[:,:,2] > 1)] = 1
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations=1)
    gradx = cv2.bitwise_and(gradx,mask)
    cv2.imshow("gradx", gradx*255)
    #cv2.line(mask, (10,10), (60,60), 255, 5)
    #print(np.rint(dst_region).astype(int))
    #cv2.fillPoly(mask, pts=[np.rint(dst_region).astype(int)], color=255)
    #cv2.imshow("mask1",mask)
    #cv2.imshow("mask",mask_and)

    

    histogram = np.sum(gradx[gradx.shape[0]//2:(5*gradx.shape[0]//6),:], axis=0)

    midpoint = int(histogram.shape[0]//2)
    leftx_base = midpoint - np.argmax(np.flip(histogram[0:midpoint]))# + quarter_point
    if histogram[leftx_base] < histo_treshold:
        left_flag=False
    else:
        left_flag=True

    rightx_base = np.argmax(histogram[midpoint:histogram.shape[0]]) + midpoint
    if histogram[rightx_base] < histo_treshold:
        right_flag=False
    else:
        right_flag=True
    
    print("leftx_base"+str(leftx_base)+"-"+str(histogram[leftx_base])+"-"+"rightx_base"+str(leftx_base)+"-"+str(histogram[rightx_base]))
    
    leftpeak = midpoint - anchor_Vehicle
    rightpeak = midpoint + anchor_Vehicle
    #if leftx_base > leftpeak or rightx_base < rightpeak:
    if leftx_base > leftpeak or rightx_base < rightpeak:
        car_out=1
    else:
        car_out=0   

    if left_flag and right_flag and laneline_obj is None:# Crear filtro de kalman
        laneline_obj = lanelines(leftx_base,rightx_base,5)
        print("Create kalman")

    if not(laneline_obj is None):
        print("realiza kalman")
        laneline_obj.age_one()
        auxValue = laneline_obj.prediction()
        if right_flag and left_flag:#(leftx_base-auxValue[0]<10) and (rightx_base-auxValue[1]<10) :# distancia entre puntos es peque;o
            print("update - kalman")
            auxValue = laneline_obj.updateCoords(leftx_base,rightx_base)
        else:# se pierde la distancia entre puntos 
            print("predic - kalman")
            auxValue =laneline_obj.predict()
        if laneline_obj.timedOut():
            laneline_obj = None
        #print("kalman")
        print(auxValue)
        

        
    #print("base"+str(histo_treshold)+" medido - rig:" + str(histogram[rightx_base]) +" medido - left:" + str(histogram[leftx_base]))
    #print("left"+str(leftx_base)+" >" + str(leftpeak) +" right" + str(rightx_base)+" <" + str(rightpeak))

    l_fit = [leftx_base,leftx_base,leftx_base]
    r_fit = [rightx_base,rightx_base,rightx_base]

    exampleImg_out1 = draw_lane(image, gradx, l_fit, r_fit, Minv)
    
    cv2.imshow("input video22", exampleImg_out1)







    #combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 255
    combined=gradx

    return combined

#video = cv2.VideoCapture(".//data//my_video-5.mkv")
video = cv2.VideoCapture(".//data//my_video-8.mp4")
while True:
    time.sleep(0.1)
    success, img = video.read()
    #print(video.isOpened())
    #print(success)
    if success:
        #print(img.shape)
        img1 = process_image(img)

        cv2.waitKey(0)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
