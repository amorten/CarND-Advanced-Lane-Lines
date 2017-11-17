## Advanced Lane Finding

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images taken from camera mounted above a car.
* Use color transforms, gradients, etc., to create a thresholded binary image that represent candidate lane line pixels.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The Files
---

[notebook]: ./advanced_lane_finding_pipeline.py "IPython notebook"
[lanefind.py]: ./lanefind.py "lanefind.py"

A detailed description of this project is given in the [write-up](writup.md).  

As discussed in the [write-up](writeup.md), the code to run this project is found in the IPython notebook [advanced\_lane\_finding\_pipeline.py][notebook], which in turn relies heavily on the module defined in [lanefind.py][].

The original README provided by Udacity has been renamed [README\_Udacity.md]().

Feel free to skip all the boring stuff described above and instead watch [the video](./lane_lines_n3.mp4) output of my lane-finding pipeline!