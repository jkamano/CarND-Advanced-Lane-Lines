## Advanced Lane Finding Project


[//]: # (Image References)

[image1]: ./report/undistorted.png "Undistorted"
[image2]: ./report/undistorted1.png "Undistorted"

[image3]: ./report/AdvanceLane.png "Pipeline"
[image4]: ./report/Schannel.png "S channel color filter"
[image5]: ./report/perspectiveStraightlines1.png "Perspective1"
[image6]: ./report/perspectiveStraightlines2.png "Perspective2"
[image7]: ./report/result.png "Output"


### The full project implementation is contained in the class *LaneDetection()*

### 1) Camera Calibration

The code to calibrate camera using chessboard pictures is in the method `calibrateCamera()`
This method receives the location of the pictures with the chessboard and computes the camera matrix and distortion coefficients.

The correction of the distortion is then implemented in the method `undist()` applying the correction for the input picture
Following figure shows correction applied to an example chessboard.

![alt text][image1]

### Pipeline (single images)
The following image represents the pipeline implemented.

![alt text][image3]

#### 1. Provide an example of a distortion-corrected image. A->(undistord)->B

Here a distorted image, it is slightly noticeable that the white car is closer to the edge on the undistorded image and that the tree look to be further out of the picture.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All threholding and color transformation is done within the method 
`getLanes()` which call separate methods for specific grandient or color analysis.
By experimentation it was decided to follow with filtering by amplitude of Sobel gradients (kernel size 7), and with threholds applied to the S channel of HLS color transformed picture whih provides great independence to the scene ilumination and shadows, as seen in following picture.

![alt text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform is setup in `setPerspective()` method.
Points were hand selected to transform the visible trappezoidal shaped lane into a rectangular shape, like it would have if seen from above.
The following picture shows the result of applying the `perpectiveTransform()` to the image. This last method uses the matrix found by the `setPerspective()` method.

Altough the points for the lane are not fixed, the perspective transformation will still be a very good approximation as long as the street doesn't have big slopes.

The following pictures show the perpective transform applied with red polygons on top whos vertices correspond to the source and destination points given.

![alt text][image5]
![alt text][image6]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I took the example given in the class that makes an histogram of the bottom half region of the transformed picture and applies a threhold to find out what are the regions of the lanes.
Knowing roughly where the lanes start on the figure, a sequence of windows with a certain margin is extended from bottom to top of the image.
All active pixels inside a window are accounted to compute a centroid for this window. The following window will have the same centroid as the previous unless adapted by the active points inside.

Active pixels, in this context, mean pixels indentified as possible lane line by the gradient+color filter.

The centroid of the active pixels inside each window will be used to fit a polynomial.

This is done for both lane lines, giving 2 2nd order polynomia.
 Two sets of polynomia are computed, one in pixel scale and other in meter scale. Using the scales also given on the class.
 
 This is done inside the method `sliding_window()`.
 
 It returns 2 polynomial one for left and other for right lane + some metadata, e.g. the number of active pixels with the objective of computing some measure of confidence in the measurement. (not implemented) 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is computed with method `getCurvature()` which returns the average of curvature of both lanes.
It computes the curvature using suggested method. It computes it for a point close to vehicle and then averages it for both lanes.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The full treatment to a frame/image is done `process_frame()` 

The final scene shows the perspective transformed identified lane on the top right corned. Shows the curvature and offset from the center on the top left corner and an overlayed lane identification on the lane in front. 

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There's several issues with my pipeline.
1) Extraction of lines, under severe light conditions or when the color of the road does not facilitate line detection, there's a clear difficulty.
Something like a normalization of light intensity or equilibrium or white balance could maybe help.

2) Low pass filter: Low passing the coefficients of the polynomial is probably not the best idea. A buffer with the older points found on previous iterations could be better. Also a smarter way to filter like a Kalman filter could be used.
The Kalman could keep the polynomial coefficients as states and measurements but their variance would depend on the error of fitting the polynomial to the centroids.

3) Take advantage of latest polynomial to remove unwanted pixels. Keeping a region of interest just around a margin of the latest polynomial.
This would help when worst conditions of the road arise.


