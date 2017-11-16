
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


### Define a class to receive the characteristics of each line detection
class Lane():
    def __init__(self):

        # Whether or not each lane line passes the validation routine
        self.left_detected = False
        self.right_detected = False

        # (x,y) positions of lane pixels
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None
        
        # fit coefficients
        self.left_fit = None
        self.right_fit = None
        # fit coefficients for corrected dimensional (x,y) coordinates
        self.left_fit_cr = None
        self.right_fit_cr = None
        
        # (x,y) values of the fit lines
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None

        # summary lane info calculated from the fit curves
        self.left_curverad = None
        self.right_curverad = None
        self.curveature_ave = None
        self.lane_offset = None
        self.left_pos_x = None
        self.right_pos_x = None
        self.lane_width = None

        # summary lane info, as above, but averaged over previous lane lines
        self.best_left_fit = None
        self.best_right_fit = None
        self.best_left_fitx = None
        self.best_right_fitx = None
        self.best_left_curverad = None
        self.best_right_curverad = None
        self.best_lane_offset = None
        self.best_lane_width = None


### Finds callibration corners given a list of image filenames.
def find_calibration_corners(img_fnames,nx,ny):

    # Prepare object points
    objp = np.zeros((nx*ny,3),np.float32) # Initizialize to zero
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # Set x and y coordinates

    imgpoints = []
    objpoints = []
    
    cal_images = []

    for fname in img_fnames:
        #print(fname)
        
        img = mpimg.imread(fname)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        cal_images.append(img)

    return (objpoints, imgpoints, cal_images) # Returns list of images with corners drawn


### Calibrates camera given a list of image filenames.
def calibrate_camera(img_fnames,nx,ny):

    objpoints, imgpoints, cal_images = find_calibration_corners(img_fnames,nx,ny)

    # Compute the camera calibration matrix and distortion coefficients
    img = cal_images[0][:,:,0] # select a single channel so that img_dims has two dimensions
    img_dims = img.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_dims, None, None)

    return (ret, mtx, dist, rvecs, tvecs, cal_images)


### Use color transforms, gradients, etc., to create a
### thresholded binary image.
### Note: the pipeline actually uses an improved version of this function
### defined next called threshold_image_improved().
def threshold_image(img, s_thresh=(170, 255), sx_thresh=(20, 100), r_thresh=(200,255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # R channel threshold
    R = img[:,:,0]
    r_binary = np.zeros_like(R)
    r_binary[(R > r_thresh[0]) & (R <= r_thresh[1])] = 1
    
    # Make an image of zeros
    zeros = np.zeros_like(img[:,:,0])
    
    # Stack each channel
    color_binary = np.dstack((r_binary,s_binary, zeros)) * 255
    
    combined_binary = np.zeros_like(img[:,:,0])
    combined_binary[(color_binary[:,:,0]==255)
                    | (color_binary[:,:,1]==255)
                    | (color_binary[:,:,2]==255)] = 1
                    
    return (color_binary, combined_binary)


### Use color transforms, gradients, etc., to create a
### thresholded binary image.
### Note: the pipeline uses this version.
def threshold_image_improved(img, sx_thresh=(12,255), sy_thresh=(5,255), s_thresh=(100,255), v_thresh=(50,255)):
    
    img = np.copy(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    v_channel = hsv[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Sobel y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in y
    abs_sobely = np.absolute(sobely) # Absolute y derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # Threshold S color channel
    s_binary = np.zeros_like(img[:,:,0])
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold V color channel
    v_binary = np.zeros_like(img[:,:,0])
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    
    # Threshold x gradient
    sx_binary = np.zeros_like(img[:,:,0])
    sx_binary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1
    
    # Threshold y gradient
    sy_binary = np.zeros_like(img[:,:,0])
    sy_binary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1
    
    
    # Make an image of zeros
    zeros = np.zeros_like(img[:,:,0])
    
    # Stack each channel
    color_binary = np.dstack((s_binary,v_binary, sx_binary & sy_binary)) * 255
    
    combined_binary = np.zeros_like(img[:,:,0])
    combined_binary[((sx_binary==1) & (sy_binary==1)) | ((s_binary==1) & (v_binary==1))] = 1
    
    return (color_binary, combined_binary)





### Finds lanes using the histogram method written by Udacity
def find_lanes_by_histogram(binary_warped,initial_bottom_frac=0.5,nwindows=9):



    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]*initial_bottom_frac):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        ##cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        ##              (0,255,0), 2)
        ##cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        ##            (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
          leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
          rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    
    lane = Lane()
    lane.leftx = leftx
    lane.rightx = rightx
    lane.lefty = lefty
    lane.righty = righty
    lane.ploty = ploty
    lane.left_fitx = left_fitx
    lane.right_fitx = right_fitx
    lane.left_fit = left_fit
    lane.right_fit = right_fit

    return (lane,out_img)

### Finds lanes based on the position of the previous lane.
### Only considers potential lane pixels within a certain
### distance from the previous lane line.
def find_lane_lanes_given_previous(binary_warped,prev_lane):

    left_fit = prev_lane.best_left_fit
    right_fit = prev_lane.best_right_fit

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                         left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    lane = Lane()
    lane.leftx = leftx
    lane.rightx = rightx
    lane.lefty = lefty
    lane.righty = righty
    lane.ploty = ploty
    lane.left_fitx = left_fitx
    lane.right_fitx = right_fitx
    lane.left_fit = left_fit
    lane.right_fit = right_fit

    return (lane,out_img)


### Draws the lane on an undistorted image
def draw_lane_lines(undist,Minv,lane,averaged_lanes=True):

    if (averaged_lanes == True):
    
        left_fitx = lane.best_left_fitx
        right_fitx = lane.best_right_fitx
        ploty = lane.ploty
    
    else: # Don't use averaged lanes. Instead use raw lane fits, even if invalid.
    
        left_fitx = lane.left_fitx
        right_fitx = lane.right_fitx
        ploty = lane.ploty
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undist[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


### Calculates lane features, such as curvature, position, and width
def calc_lane_features(lane,img_shape):

    leftx = lane.leftx
    rightx = lane.rightx
    lefty = lane.lefty
    righty = lane.righty
    ploty = lane.ploty
    
    # Determine the curvature of the lane and vehicle position
    # with respect to center.

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print('Left curvature: ',left_curverad, 'm')
    #print('Right curvature: ',right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    # Calculate x position of the middle of the image
    image_center_x = img_shape[0]/2 * xm_per_pix
    # Calculate x positions of the two lanes
    y_eval_cr = y_eval*ym_per_pix
    left_pos_x = left_fit_cr[0]*y_eval_cr**2 + left_fit_cr[1]*y_eval_cr + left_fit_cr[2]
    right_pos_x = right_fit_cr[0]*y_eval_cr**2 + right_fit_cr[1]*y_eval_cr + right_fit_cr[2]
    # Calculate center offset
    lane_offset = (left_pos_x + right_pos_x)/2 - image_center_x
    # Print the above values
    #print('Offset: ',lane_offset,'m')
    #print('Left lane pos: ',left_pos_x,'m')
    #print('Center pos: ',image_center_x,'m')
    #print('Right lane pos: ',right_pos_x,'m')

    lane.left_fit_cr = left_fit_cr
    lane.right_fit_cr = right_fit_cr

    lane.left_curverad = left_curverad
    lane.right_curverad = right_curverad
    lane.curveature_ave = (left_curverad + right_curverad) / 2.0
    lane.lane_offset = lane_offset
    lane.left_pos_x = left_pos_x
    lane.right_pos_x = right_pos_x
    lane.lane_width = right_pos_x - left_pos_x

    return lane


### Checks whether a lane has valid fit parameters.
### The fit parameters are tested to see if they lie within a presribed range,
### and the difference with the previous lane parameters are also tested.
### Lane widths are also checked.
def check_lane_validity(lane,prev_lane,
                        left_curverad_range=[0,1000],right_curverad_range=[0,1000],
                        left_pos_x_range=[-100,100],right_pos_x_range=[-100,100],
                        lane_width_range=[0,100],
                        left_curverad_max_ratio=1000,right_curverad_max_ratio=1000,
                        left_pos_x_max_diff=100,right_pos_x_max_diff=100,
                        lane_width_max_diff=100):


    if (prev_lane == None):
        # If there is no previous lane, then just make it the same as current lane
        # Then comparisons witht he current lane will always succeed.
        prev_lane = lane

    # Check fit parameters for current lane lie within prescribed ranges
    if ( left_curverad_range[0] < lane.left_curverad < left_curverad_range[1]
        and left_pos_x_range[0] < lane.left_pos_x < left_pos_x_range[1]
        and (1./left_curverad_max_ratio) < np.abs(lane.left_curverad/prev_lane.left_curverad) < left_curverad_max_ratio
        and np.abs(lane.left_pos_x - prev_lane.left_pos_x) < left_pos_x_max_diff ):
        
        lane.left_detected = True
        
    # Check fit parameters are not too different from previous lane fit parameters
    if ( right_curverad_range[0] < lane.right_curverad < right_curverad_range[1]
            and right_pos_x_range[0] < lane.right_pos_x < right_pos_x_range[1]
            and (1./right_curverad_max_ratio) < np.abs(lane.right_curverad/prev_lane.right_curverad) < right_curverad_max_ratio
            and np.abs(lane.right_pos_x - prev_lane.right_pos_x) < right_pos_x_max_diff ):
            
        lane.right_detected = True

    # Check that the lane width is reasonable.
    if not( lane_width_range[0] < lane.lane_width < lane_width_range[1]
           and  np.abs(lane.lane_width - prev_lane.lane_width) < lane_width_max_diff ):

        lane.left_detected = False
        lane.right_detected = False

    # Prints out some of the fit parameters
    print("Curve: ",lane.left_curverad, "m,", lane.right_curverad, "m" )
    print("Pos: ",lane.left_pos_x, "m,", lane.right_pos_x, "m")
    print("Width: ",lane.lane_width, "m" )
    print("Offset: ", lane.lane_offset, "m")

    return (lane.left_detected, lane.right_detected)


### Calculate the "best" lane fit coefficients.
### Here, "best" means the values have been appropriately averaged
### over several previous lane calculations.
def calc_best_lane(recent_lanes):

    # Assumes that recent_lanes is a non-empty list of lanes,
    # with recent_lanes[0] being the most recent.
    # This function modifies the Lane stored in recent_lanes[0],
    # updating it with the calculated best lane statistics.
    # Returns recent_lane[0] after the update.
    
    
    # For simplicity, instead of calculating the best fit based on previous lane raw data,
    # we calculate the best fit based on previous lane best fit data.
    
    lane = recent_lanes[0]

    if (lane.left_detected == True and lane.right_detected == True): # If the current lanes are valid
        print("Current lanes are valid.")
        # Temporarily set current lane best fit values to raw fit values.
        # Later we will average the current best fit with previous best fit
        lane.best_left_fit = lane.left_fit
        lane.best_right_fit = lane.right_fit
        lane.best_left_curverad = lane.left_curverad
        lane.best_right_curverad = lane.right_curverad
        lane.best_lane_offset = lane.lane_offset
        lane.best_lane_width = lane.lane_width
            
    else: # If the current lanes are not valid
        print("Current lanes were not valid.")
        if (len(recent_lanes) < 2):
            print("Uh oh! The lane lines were not detected in the very first frame! The pipeline does not yet handle that possibility!")
        else: # There is at least one previous lane saved in recent_lanes
            prev_lane = recent_lanes[1]
        
        lane.best_left_fit = prev_lane.best_left_fit
        lane.best_right_fit = prev_lane.best_right_fit
        lane.best_left_curverad = prev_lane.best_left_curverad
        lane.best_right_curverad = prev_lane.best_right_curverad
        lane.best_lane_offset = prev_lane.best_lane_offset
        lane.best_lane_width = prev_lane.best_lane_width

    # Now it's time to average over the previous n best_fit values to
    # obatin the new best_fit values. The average is weighted by wts.

    n = len(recent_lanes)
    wts = np.linspace(n,1,n) # Use higher weights for most recent images, w/ wts decreasing to zero linearly
    
    lane.best_left_fit = np.average([l.best_left_fit for l in recent_lanes],weights=wts,axis=0)
    lane.best_right_fit = np.average([l.best_right_fit for l in recent_lanes],weights=wts,axis=0)
    #print([l.best_left_curverad for l in recent_lanes])
    lane.best_left_curverad = np.average([l.best_left_curverad for l in recent_lanes],weights=wts,axis=0)
    #print([l.best_left_curverad for l in recent_lanes])
    lane.best_right_curverad = np.average([l.best_right_curverad for l in recent_lanes],weights=wts,axis=0)
    lane.best_lane_offset = np.average([l.best_lane_offset for l in recent_lanes],weights=wts,axis=0)
    lane.best_lane_width = np.average([l.best_lane_width for l in recent_lanes],weights=wts,axis=0)

    left_fit = lane.best_left_fit
    right_fit = lane.best_right_fit
    ploty = lane.ploty
    lane.best_left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    lane.best_right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return lane






