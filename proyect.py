import numpy as np
import cv2
import math
import random
import time


histo_treshold=50
leftpeak=129
rightpeak=440
leftx_base=0
rightx_base=0
left_flag=0
right_flag=0
car_out=0

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def abs_sobel_thresh(img, orient='x',sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    #binary_output = ((gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])) =1
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv



def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    #print(ploty)
    #left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    #right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
    left_fitx = np.ones_like(ploty) * l_fit[0]
    right_fitx = np.ones_like(ploty) * r_fit[0]
    #print(left_fitx)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    if car_out == 0:
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (0,0, 255))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


def process_image(image):
    global leftx_base, rightx_base, left_flag, right_flag, car_out
    #image = gaussian_blur(image, 5)
    imshape = image.shape
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    poly_int=[(0, round(imshape[0]*0.9)),
            (round(imshape[1]*0.36), round(imshape[0]*0.5)),
            (round(imshape[1]*0.60), round(imshape[0]*0.5)),
            (imshape[1], round(imshape[0]*0.9))]
    poly_out=[(0+100,imshape[0]),
              (0+100, 0),
              (imshape[1]-100, 0),
              (imshape[1]-100, imshape[0])]
    regionFindingLane = np.array(poly_int, dtype=np.int32)
    src_region = np.array(poly_int, dtype=np.float32)
    dst_region =  np.array(poly_out, dtype=np.float32)
    #imgInterest = region_of_interest(edges, regionFindingLane)

    img_unwarp, M, Minv = unwarp(image, src_region, dst_region)
    cv2.imshow("unwrap", img_unwarp)

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img_unwarp, orient='x', sobel_kernel=ksize, thresh=(50, 255))# 0 a 255
    #grady = abs_sobel_thresh(img_unwarp, orient='y', sobel_kernel=ksize, thresh=(50, 100))# 0 a 255
    #mag_binary = mag_thresh(img_unwarp, sobel_kernel=ksize, mag_thresh=(50, 255))# 0 a 255
    #dir_binary = dir_threshold(img_unwarp, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))# 0 a pi/2
    s_binary = hls_select(img_unwarp, thresh=(30, 255))
    color_binary = np.dstack(( np.zeros_like(gradx), gradx, s_binary)) * 255
    cv2.polylines(image, [regionFindingLane], True, (255, 255, 0), 1)
    cv2.imshow("gradx", gradx*255)
    #cv2.imshow("hls", s_binary*255)
    cv2.imshow("input video video", image)
    histogram = np.sum(gradx[gradx.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    #quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[0:midpoint])# + quarter_point
    if histogram[leftx_base] < histo_treshold:
        left_flag=False
    else:
        left_flag=True

    rightx_base = np.argmax(histogram[midpoint:histogram.shape[0]]) + midpoint
    if histogram[rightx_base] < histo_treshold:
        right_flag=False
    else:
        right_flag=True
    #print("leftpeak")
    #print(leftx_base)
    #print("rightpeak")
    #print(rightx_base)
    l_fit = [leftx_base,leftx_base,leftx_base]
    r_fit = [rightx_base,rightx_base,rightx_base]

    exampleImg_out1 = draw_lane(image, gradx, l_fit, r_fit, Minv)
    cv2.imshow("input video22", exampleImg_out1)

    if leftx_base > leftpeak or rightx_base < rightpeak:
        car_out=1
    else:
        car_out=0





    #combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 255
    combined=gradx


    
    return combined

video = cv2.VideoCapture(".//my_video-5.mkv")
while True:
    time.sleep(0.1)
    success, img = video.read()
    #print(video.isOpened())
    #print(success)
    if success:
        #print(img.shape)
        img1 = process_image(img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
