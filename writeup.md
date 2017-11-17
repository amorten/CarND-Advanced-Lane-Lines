## Writeup 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibrate]: ./output_images/undistort_example.jpg "Undistorted"
[undistorted]: ./output_images/undistorted_test_image.jpg "Road Transformed"
[binary]: ./output_images/binary.jpg "Binary Examples"
[birds_eye_image]: ./output_images/birds_eye_image.jpg "Warp Example"
[birds_eye_binary]: ./output_images/birds_eye_binary.jpg "Warp Example"

[lane_fit]: ./output_images/lane_fit.jpg "Fit Visual"
[lane_area]: ./output_images/lane_area.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[notebook]: ./advanced_lane_finding_pipeline.py "IPython notebook"
[lanefind.py]: ./lanefind.py "lanefind.py"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in code cell #3 of the IPython notebook [./advanced\_lane\_finding\_pipeline.ipynb][notebook]. It relies on the functions `lanefind.find_calibration_corners` and `lanefind.calibrate_camera` defined in the module [lanefind.py][] on lines #54-100. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to all of the calibration images using the `cv2.undistort()`. All of the undisorted images are plotted in the [IPython notebook][notebook]. Here is an example result: 

![undistorted calibrtion image][calibrate]

### Pipeline (test images)

#### 1. Provide an example of a distortion-corrected image.

I apply distortion correction to a test image in the [notebook][notebook] in cell #4 using the function `cv2.undistort`. The `cv2.undisort` function takes in the test image and some of the distortion correction matrices output by my `lanefind.calibrate_camera` function.

Here is an example undistorted test image:

![example undistorted test image][undistorted]

The distortion correction is applied to all test images in cell #4 of the [notebook][].

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. I do so for all test images in cell #5 of the [notebook][]. The code relies on the function `lanefind.threshold_image_improved` defined in the module [lanefind.py] on lines #145-199.

Here is how the `lanefind.threshold_image_improved` function works:

To calculate gradients, I fist convert the image to grayscale. I use `cv2.Sobel` to calculate the absolute value of the gradients in both the x- and y-directions. Pixels within the image are selected as potential line pixels if the absolute values of *both* the x- and y-gradients lie within prescribed threshold ranges of [12,255] and [5,255] respectively. Those thresholds appear to work well in selecting lines that are not too horizontal.

To apply thresholds based on color, I first use `cv2.cvtColor` to convert to HLS and HSV color spaces. I select the S channel from HLS and the V channel from HSV and apply the thresholds [100,255] and [50,255] respectively. Pixels with color channel values within *both* threshold bounds are selected as potential lane line pixels.

Combining the gradient thresholds and color thresholds, image pixels are selected as lane line pixels if *either* the gradient thresholds are met *or* the color thresholds are met.

Here is an example binary image (right column) calulculated from an undistorted test image (left column):

![thresholded binary images][binary]

All of the binary images for all of the test images can be found below cell #5 of the [notebook][].

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In cell #6 of the [notebook][], I choose four points on an image (with straight lane lines) that would correspond to a rectangle if viewed from above (birds-eye-view). The four selected points are stored in `src`. I choose four destination points `dst` that form a rectangle.  I then use `cv2.getPerspectiveTransfrom` to calculate the transformation matrix `M` based on `src` and `dst`. The size and position of the rectangle defined by `dst` is chosen so that the full extent of the lane lines remain visible in all of the transformed test images. I drew the four `src` points on a test image below cell #6 of the [notebook][].

Then, in cell #7 of the [notebook][] I use `cv2.warpPerspective` to transform all of the undistorted test images as well as all of the binary images to birds-eye-view.

Here are an example transformed image and transformed binary (right column) side-by-side with the corresponding undistorted image:

![transformed to birds-eye-view][birds_eye_image]  
![transformed to birds-eye-view][birds_eye_binary]

Transformed images and transformed binaries for all test images are shown below cell #7 in the [notebook][].

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I defined two functions in [lanefind.py][] that identify lane-line pixels and fit their positions with polynomials: `find_lane_lines_by_histogram` on lines #205-305, and `find_lane_lines_given_previous` on lines #307-359.

The function `find_lane_lines_by_histogram` assumes there are no previous lane lines (from previous frames of a video) to help find the lane lines. Instead, it divides the birds-eye-view binary into left and right halves and searches each half for potential lane line pixels (for left and right lane lines, respectively). The algorithm (and most of the code) is as described in the class. Each half of the binary image is divided into nine windows in the vertical direction. Starting from the bottom of the image a histogram (with ten bins) is calculated along the x direction, and the argmax of the histogram determines the window within which to choose lane pixels. Them moving up toward the top of the image, pixels are selected that lie within a window whose x-position lies within a prescribed margin of the window below. In this way, only pixels that form a roughy continuous curve from the bottom of the image to the top of the image are selected.

The function `find_lane_lines_given_previous` assumes we already have fitted lane lanes from the previous video frame.  In this case, all pixels within a certain distance of the fitted lane lines are selected.

For both functions, after prospective lane line pixels have been selected, they are fit by a second degree polynomial.

Cell #8 of the [notebook] calculate lane lines using the histogram method and then outputs binary images with colored lane line pixels and best-fit quadratic curves. Here is an example: 

![Lane lines with fit curves drawn][lane_fit]

Similar lane line images for all test images are shown in the [notebook][] below cell #8.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature (for each lane line) and the position of the vehicle (the "lane_offset") are calculated in cell #9 of the [notebook][], using the function `lanefind.calc_lane_features` found on lines #362-416 in the module [lanefind.py][].

Basically the function `lanefind.calc_lane_features` recalcuates the quadratic fit curves for the left and right lanes after scaling the x- and y-coordinates to dimensional values (in meters). The "positions" of the lanes are calculated to be the (dimensional) x-coordinate of the lane fit curve with the bottom of the image. Similarly, the lane curvatures are calculated at the bottom of the image, using the standard formula for radius of curvature in terms of dy/dx and d^2y/dx^2. The position of the car is assumed to be the center of the image, and its offset is calculated with respect to the average position of the two lane lines (*i.e.*, the lane center).

The lane curvatures, car offset, and other lane features for all of the test images are printed out below cell #9 of the [notebook][]. For example, for the first test image, the radius of curvature for the left lane is 7988 m, the radius of curvature for the right lane is 3209 m, and the offset is -0.06 m (*i.e.*, the car is 0.06 meters to the left of the center of the lane).


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is done in cell #10 of the [notebook][], using the function `lanefind.draw_lane_lines` found in the module [lanefind.py][] on lines #419-486. The function simply uses `cv2.fillPloy` to plot the area between the lines in green based on the fit curves for the left and right lanes. 
The code can also draw the left lane in red and the right lane in blue, but I commented that part out because I think the lane lines are more precisely indicated by showing the area between the left and right lane lines in green only. The left and right edges of the green region correspond to the left and right lane lines.

The output for all images are plotted below cell #10 of [notebook][]. Here is one example:

![Example lane area plot][lane_area]

Note that negative value of -0.10 m to the *right* means that the car is actually 0.10 m to the *left* of the lane center. 

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline is defined in cell #12 of the [notebook]. As discussed below, it uses a weighted average of the three most recent successful lane detections. The pipleline is tested on the test images in cell #13, and the video is created in cell #14.

Here's a [link to my video result](./lane_lines_n3.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In addition to the steps discussed above, I also defined a Lane class in the module [lanefind.py] for storing fit data of previous lanes. I then define a global variable `recent_lanes` that the pipeline has access to. Another global variable, `n_history`, defines the number of previous lanes to average over at each frame.

Within the pipeline in cell #12 in the [notebook], I make use of the functions `lanefind.check_lane_validity` (lines #490-537 in [lanefind.py]) and `calc_best_lane` (lines #540-603 in [lanefind.py]).  For each frame, `lanefind.check_lane_validity` checks whether the lane features (such as curvature and lane_width) have reasonable values and whether they did not change too much from the previous frame. If the lane feature values appear unreasonable, then there are not included in the weighted averaging.  The weighted average is accomplished by the function `lanefind.calc_best_lane`.  The averaging is a weighted average over the most recent `n_history` lanes, with the most recent lane having the largest weight (the weights decrease linearly to zero).

The main issue I had with this project was accurately finding the lane lines when there were shadows from trees over the road. At first I tried solving the problem with averaging over previous frames, but that was only partially successful. I solved the problem by trying different ways to calculate the binary image using gradients and various color thresholds.

My pipeline will likely fail if there are other lines on the road with sharp gradients that are roughly parallel with the lane lines. Examples of such situations can be found in the challenge videos. My pipeline will also fail if the lane lines are determined to be invalid for more than `n_history` frames in a row. Failure would also occur if the car switches lanes.

I could make the pipeline more robust by also checking that the inverse radii of curvature of the two lane lines are roughly similar and that the left and right lane lines curve in the same direction (when not roughly straight). Also, when one lane line is successfully detected but the other is not, the undetected lane line could be determined based on previous calculations of the lane width. Also, when detecting lane line pixels, I calculated the gradient using grayscale images, which will pick up gradients on the road even where there is no white or yellow paint. I suspect it would have been better to apply the gradient to one or two color channels instead to somehow check for the transition from white-ish or yellow-ish to gray-ish.

