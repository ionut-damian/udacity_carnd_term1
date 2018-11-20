import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', thresh_min=5, thresh_max=100):    
    """ 
    Define a function that applies Sobel x or y, 
    then takes an absolute value and applies a threshold.
    """
    sobel = []    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #grayscale conversion
    else:
        gray = img
    
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    #scale to 0..255
    sobel = np.absolute(sobel)
    sobel = np.uint8(255*sobel/np.max(sobel))

    #apply threshold
    binary_output = (sobel > thresh_min) & (sobel < thresh_max) 
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ 
    Compute direction of gradients and apply threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #grayscale conversion
    
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    dir = np.arctan2(sobely,sobelx)

    binary_output = (dir > thresh[0]) & (dir < thresh[1]) 
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #grayscale conversion
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    mag = np.sqrt(sobelx**2 + sobely**2)    
    mag = np.uint8(255*mag/np.max(mag))

    binary_output = (mag > mag_thresh[0]) & (mag < mag_thresh[1]) 
    return binary_output

def hls_select(img, channel_id, thresh=(0, 255)):
    """ 
    Transform image to HLS, extract S channel and apply threshold
    """    
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    ch = img_hls[:,:,channel_id]
    ch = np.uint8(255.0*ch/np.max(ch))
    binary_output = np.zeros_like(ch)

    # Identify pixels below the threshold
    color_thresholds = (ch > thresh[0]) & (ch <= thresh[1])            
    binary_output[color_thresholds] = 1
    
    return binary_output