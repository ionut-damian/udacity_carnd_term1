import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob 
import cv2

import calibration as calib
import thresholds as thresh

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #polynomial coefficients for the most recent fits
        self.recent_fit = []
        #radius of curvature of the line in some units
        self.curvature_pixel = 0 
        self.curvature_real = [] 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 

    def addFit(self, fit):
        self.recent_fit.append(fit)

        if len(self.recent_fit) > fit_history_size_:
            self.recent_fit.pop(0)

    def getFitMean(self):
        return np.mean(self.recent_fit, axis=0)

    def addCurvature(self, radius):
        self.curvature_real.append(radius)

        if len(self.curvature_real) > curvature_history_size_:
            self.curvature_real.pop(0)

    def getCurvatureMean(self):
        return np.mean(self.curvature_real)

def hist(img, bottom, top):
    """
    compute histogramm of image area (expressed in percentage of whole image)
    """
    bottom_half = img[bottom:top,:]
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

def findHistogramMax(histogram, left, right, win_size=100, min_num_points=0):
    """
    analyze histogram to find maximum under certain conditions
    """    
    histmax = np.argmax(histogram[left:right])
    if histmax == 0:
        return None

    if min_num_points == 0:
        return histmax
    else:                
        result = None
        mask = np.array(np.ones_like(histogram), dtype=bool)
        histogram_masked = histogram
        mask_size = 0
        # iterate until we find a maximum which fullfills the conditions or there are no more elements left
        while mask_size < right-left | histmax == 0:
            #determine how many points reside in a window around histmax
            num = np.sum(histogram_masked[histmax-win_size//2:histmax+win_size//2])
            if num > min_num_points:
                return histmax
            else:
                mask[histmax] = False
                mask_size += 1
                histogram_masked = histogram[mask]

                plt.plot(histogram_masked)
                plt.show()

                histmax = np.argmax(histogram_masked[left:right])

        return None

def find_lane_pixels(img, startx, img_visualization):
    # Set height of windows - based on nwindows above and image shape
    shape = img.shape
    window_height = np.int(shape[0]//nwindows_)
    # Identify the x and y indices of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    x_current = startx
    # track the slope of the movement of the windows
    slope = None
    cnt_valid_win = 0 

    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows_):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        # Find the four below boundaries of the window ###
        win_x_low = x_current - window_width_//2
        win_x_high = x_current + window_width_//2
        
        # Draw the windows on the visualization image
        cv2.rectangle(img_visualization, (win_x_low, win_y_low), (win_x_high, win_y_high), (0,255,0), 2) 
                
        ### Identify the nonzero pixels in x and y within the window
        # evaluate which indices (in the nonzero array) fall within our windows
        new_inds = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)

        # extract the indices (of the nonzero array) which fall within the window
        new_inds = new_inds.nonzero()[0]
        
        # if we find enough points in a window, reposition the window
        if len(new_inds) > minpix_:
            cnt_valid_win += 1
            # comput mean X-value of all nonzero pixels in the window
            new_window_center = int(np.mean(nonzerox[new_inds]))
            #compute new slope
            new_slope = new_window_center - x_current
            # sanity check for new window center
            # skip if this is the first time we find points
            if slope == None:
                x_current = new_window_center  
                #update slope only if we already detected two valid window centers
                if cnt_valid_win > 1:
                    slope = new_slope              
            else:
                # check if the slope direction is the same the the slopes are similar
                if (abs(new_slope-slope) < slope_max_diff_):
                    slope = new_slope
                    x_current = new_window_center
                else:
                    #bad slope detected, break
                    break

        # Append these indices to the lists
        lane_inds.append(new_inds)      

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
    
    # mark pixels in img
    img_visualization[y, x] = [255, 0, 0]

    return x, y

def samplePoly2Deg(poly, y):    
    return poly[0]*y**2 + poly[1]*y + poly[2]

def update_lane_pixels(img, poly, margin):
    # Grab activated pixels
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # old line
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0]) 
    x_old = samplePoly2Deg(poly, plot_y)
    
    # identify indices in nonzero arrays which are within our window
    lane_inds = (nonzerox > x_old[nonzeroy] - margin) & (nonzerox < x_old[nonzeroy] + margin)

    # Again, extract line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 

    return x, y

def measure_curvature(ploty, left_fit, right_fit, scale_y=1):    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty) * scale_y
    
    left_curverad = rcurveabs(left_fit, y_eval)
    right_curverad = rcurveabs(right_fit, y_eval)
    
    return left_curverad, right_curverad

def rcurve(fit, y):
    return ((1 + (2*fit[0]*y + fit[1])**2)**(3/2))/(2*fit[0])

def rcurveabs(fit, y):
    return ((1 + (2*fit[0]*y + fit[1])**2)**(3/2))/abs(2*fit[0])

def process_image(img):  
    """
    Find lane lines in a single image and draw them on the image
    """          
    shape = img.shape
    plot_y = np.linspace(0, shape[0]-1, shape[0]) 

    if save_img_:
        global frame_
        frame_ += 1    
        cv2.imwrite("output_images/" + video_name_ + str(frame_) + ".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    #undistort image
    img_undist = calib.undistort(img, mtx_, dist_)

    if save_img_:
        cv2.imwrite("output_images/" + video_name_ + str(frame_) + "_undist.jpg", cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))

    #find edges
    gradx_binary = thresh.abs_sobel_thresh(img_undist[:,:,0], orient='x', thresh_min=20, thresh_max=100) #gradient on blue channel
    dir_binary = thresh.dir_threshold(img_undist, sobel_kernel=25, thresh=(0.8, 1.2)) #gradient direction
    sat_binary = thresh.hls_select(img_undist, 2, thresh=(80, 255)) #saturation selector
    #mag_binary = thresh.mag_thresh(img_undist, sobel_kernel=3, mag_thresh=(30, 100))

    red = img_undist[:,:,0]
    red_binary = np.zeros_like(red)
    red_half = red[red.shape[0]//2:,:]
    red_max = np.max(red_half)
    red_mean = np.mean(red_half)
    red_thresh = (red_max - red_mean)/2 + red_mean
    red_binary[red > red_thresh] = 1

    #img_thresholded = gradx_binary | (sat_binary & dir_binary)
    #img_thresholded = (gradx_binary | sat_binary) & dir_binary
    #img_thresholded = (gradx_binary | mag_binary) & ((sat_binary | red_binary) & dir_binary)
    img_thresholded = (gradx_binary | sat_binary | red_binary) & dir_binary

    if save_img_:
        cv2.imwrite("output_images/" + video_name_ + str(frame_) + "_masked.jpg", img_thresholded*255)

    #tranform perspective to birds-eye
    M, M_inv = calib.computePerspectiveTransform(img_undist, draw_roi_)
    img_warp = calib.warp(img_thresholded, M)
    
    if save_img_:
        cv2.imwrite("output_images/" + video_name_ + str(frame_) + "_warp.jpg", img_warp*255)

    # Take a histogram of the bottom half of the image
    histogram = hist(img_warp, shape[0]//2, shape[0])
    midpoint = np.int(histogram.shape[0]//2)
    
    # Create an output image to draw on and visualize the result
    img_visualization = np.dstack((img_warp, img_warp, img_warp)) * 255

    left_reset = False
    right_reset = False

    # Find all pixels which belong the the left and right lane lines  
    if left_.detected == False:
        # find starting point using histogram
        start = np.argmax(histogram[:midpoint])
        # start sliding window algorithm to find points        
        left_.allx, left_.ally = find_lane_pixels(img_warp, start, img_visualization)
        left_reset = True
    else:
        #update points
        left_.allx, left_.ally = update_lane_pixels(img_warp, left_.getFitMean(), window_width_//2)

    if right_.detected == False:
        # find starting point using histogram
        start = np.argmax(histogram[midpoint:]) + midpoint
        # start sliding window algorithm to find points        
        right_.allx, right_.ally = find_lane_pixels(img_warp, start, img_visualization)
        right_reset = True
    else:
        #update points
        right_.allx, right_.ally = update_lane_pixels(img_warp, right_.getFitMean(), window_width_//2)    

    # compute polynomials
    if len(left_.allx) > minpix_:        
        left_.current_fit = np.polyfit(left_.ally, left_.allx, 2)
        left_.curvature_pixel = np.min(rcurve(left_.current_fit, plot_y))
        # check if curve is plausible
        if abs(left_.curvature_pixel) > curvature_min_:
            left_.detected = True
            left_.addFit(left_.current_fit)

    if len(right_.allx) > minpix_:        
        right_.current_fit = np.polyfit(right_.ally, right_.allx, 2)
        right_.curvature_pixel = np.min(rcurve(right_.current_fit, plot_y))
        # check if curve is plausible
        if abs(right_.curvature_pixel) > curvature_min_:
            right_.detected = True
            right_.addFit(right_.current_fit)

    # check if curve matches between lines
    left_curve_rating = 10000 / left_.curvature_pixel
    right_curve_rating = 10000 / right_.curvature_pixel
    if left_.detected & right_.detected & (abs(left_curve_rating - right_curve_rating) > curvature_max_diff_):
        left_.detected = False
        right_.detected = False

    # sample polynomes
    left_x_mean = None
    right_x_mean = None
    if len(left_.recent_fit) > 0:
        left_x_mean = samplePoly2Deg(left_.getFitMean(), plot_y)
    if len(right_.recent_fit) > 0:
        right_x_mean = samplePoly2Deg(right_.getFitMean(), plot_y)
    left_x = samplePoly2Deg(left_.current_fit, plot_y)
    right_x = samplePoly2Deg(right_.current_fit, plot_y)  

    if save_img_:
        imglines = np.copy(img_visualization)
        for i in range(0, len(plot_y)-1):
            cv2.line(imglines, (int(left_x[i]), int(plot_y[i+1])), (int(left_x[i]), int(plot_y[i+1])), (255,0,0), 4)
            cv2.line(imglines, (int(right_x[i]), int(plot_y[i+1])), (int(right_x[i]), int(plot_y[i+1])), (255,0,0), 4)
        cv2.imwrite("output_images/" + video_name_ + str(frame_) + "_lines.jpg", imglines)

    if left_.detected & right_.detected:        
        #check if distance between lines is plausible
        dist = np.mean(abs(left_x - right_x)) 
        if (dist < 3.0 / xm_per_pix_) | (dist > 4.5 / xm_per_pix_):
            left_.detected = False
            right_.detected = False

    if show_process_interim_:        
        f, axarr = plt.subplots(2,3, figsize=(24, 9))
        f.tight_layout()
        
        axarr[0,0].set_title('Original Image')
        axarr[0,0].imshow(img)      
        
        axarr[0,1].set_title('Thresholded Image')
        axarr[0,1].imshow(img_thresholded)
        
        axarr[0,2].set_title('Warped Image')
        axarr[0,2].imshow(img_warp)
        
        axarr[1,0].set_title('Histogram')
        axarr[1,0].plot(histogram)

        ######################
        ## Lines
        axarr[1,1].set_title('Lines')             
        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(img_visualization)
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = window_width_//2

        axarr[1,1].plot(left_x, plot_y, color='yellow')
        axarr[1,1].plot(right_x, plot_y, color='yellow')

        if left_reset == False:
            left_line_window1 = np.array([np.transpose(np.vstack([left_x - margin, plot_y]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_x + margin, plot_y])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        
        if right_reset == False:
            right_line_window1 = np.array([np.transpose(np.vstack([right_x - margin, plot_y]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_x + margin, plot_y])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        # Draw the lane onto the warped blank image
        result = cv2.addWeighted(img_visualization, 1, window_img, 0.3, 0)
        axarr[1,1].imshow(result)     
        
        ######################
        ## Lane
        axarr[1,2].set_title('Lane')                
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img_warp).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        draw_weight = 0.5
        # Recast the x and y points into usable format for cv2.fillPoly() 
        if (left_x_mean != None) & (right_x_mean != None):
            pts_left = np.array([np.transpose(np.vstack([left_x_mean, plot_y]))])        
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x_mean, plot_y])))])
            pts = np.hstack((pts_left, pts_right))
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        for i in range(0, len(plot_y)-1):
            cv2.line(color_warp, (int(left_x[i]), int(plot_y[i+1])), (int(left_x[i]), int(plot_y[i+1])), (255,0,0), 4)
        
        for i in range(0, len(plot_y)-1):
            cv2.line(color_warp, (int(right_x[i]), int(plot_y[i+1])), (int(right_x[i]), int(plot_y[i+1])), (255,0,0), 4)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = calib.warp(color_warp, M_inv)
        # Combine the result with the original image
        result = cv2.addWeighted(img_undist, 1, newwarp, draw_weight, 0)
        # draw curvature value
        cv2.putText(result, "curvature left=" + str(left_curve_rating), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(result, "curvature right=" + str(right_curve_rating), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        
        # output state of line detection
        cv2.putText(result, "|", (1100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        if left_.detected:
            cv2.putText(result, "left", (1040,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        if right_.detected:
            cv2.putText(result, "right", (1120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        axarr[1,2].imshow(result)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        
    ## Draw lane onto image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    draw_weight = 0.3
    # compute curvature (in meters)
    left_fit_real = np.polyfit(left_.ally * ym_per_pix_, left_.allx * xm_per_pix_, 2)
    right_fit_real = np.polyfit(right_.ally * ym_per_pix_, right_.allx * xm_per_pix_, 2)
    curv = measure_curvature(plot_y, left_fit_real, right_fit_real, ym_per_pix_)
    left_.addCurvature(curv[0])
    right_.addCurvature(curv[1])

    if (left_x_mean != None) & (right_x_mean != None):
        # Recast the x and y points into usable format for cv2.fillPoly() 
        pts_left = np.array([np.transpose(np.vstack([left_x_mean, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x_mean, plot_y])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = calib.warp(color_warp, M_inv)
    # Combine the result with the original image
    result = cv2.addWeighted(img_undist, 1, newwarp, draw_weight, 0)
    # draw curvature value
    cv2.putText(result, "curvature=" + str(np.mean((left_.getCurvatureMean(), right_.getCurvatureMean()))), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    # compute offset
    offset = (img.shape[1]/2 - (right_x_mean[-1] - left_x_mean[-1])) * xm_per_pix_
    cv2.putText(result, "offset=" + str(offset), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    # output state of line detection
    cv2.putText(result, "|", (1100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    if left_.detected:
        cv2.putText(result, "left", (1040,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    if right_.detected:
        cv2.putText(result, "right", (1120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        
    if save_img_:
        cv2.imwrite("output_images/" + video_name_ + str(frame_) + "_final.jpg", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result

def process_video(file, output_dir, length=-1):
    #load video
    clip1 = VideoFileClip(file)
    if length > 0:
        clip1 = clip1.subclip(0,length)

    #process video frame by frame
    global frame_, video_name_
    video_name_ = file[:file.find(".")]
    frame_ = 0
    out_clip = clip1.fl_image(process_image)

    #write result to file
    out_clip.write_videofile(output_dir +"/"+ file, audio=False)

    out_clip.reader.close()
    out_clip.audio.reader.close_proc()

##################
## MAIN PROGRAM ##
##################

## CONSTANTS ##
# Define conversions in x and y from pixels space to meters
ym_per_pix_ = 30/720 # meters per pixel in y dimension
xm_per_pix_ = 3.7/700 # meters per pixel in x dimension

## HYPERPARAMETERS ##
nwindows_ = 15 #9 # Choose the number of sliding windows
window_width_ = 200 # Set the width of the windows
minpix_ = 80 #100 # Set minimum number of pixels found to recenter window

## OPTIONS ##
show_calib_result_ = False
show_process_interim_ = False
draw_roi_ = False
save_img_ = True
curvature_history_size_ = 20
fit_history_size_ = 20
curvature_min_ = 500
curvature_max_diff_ = 7
slope_max_diff_ = 1280 * 0.07 #2% of max width


# Do calibration
chessboard = glob.glob('camera_cal\\calibration*.jpg')
mtx_, dist_ = calib.computeCalibMatrix(chessboard, 9, 6, show_calib_result_)

left_ = Line()
right_ = Line()

frame_ = 0
video_name_ = ""

testimages = glob.glob('output_images\\video12*.jpg')
for fname in testimages:
    left_ = Line()
    right_ = Line()
    #process_image(mpimg.imread(fname))


#process_image(mpimg.imread("test_images\\straight_lines1.jpg"))
#process_image(mpimg.imread("output_images\\challenge_video1.jpg"))

process_video("project_video_full.mp4", "output_video/")

left_ = Line()
right_ = Line()
process_video("challenge_video_full.mp4", "output_video/")
#process_video("harder_challenge_video.mp4", "output_video/", 5)