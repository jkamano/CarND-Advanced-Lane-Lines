#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:33:20 2017

@author: joao.ferreira
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



class LaneDetection():
    def __init__(self):
        """ Initialize LaneDetection module
        """
        self.mtx = []
        self.dist= []
        self.pSource = []
        self.pDest = []
        self.Mperspective = []
        #Assumption for meter/pixel
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.polyLeft  = []
        self.polyLeft_px  = []
        self.confLeft  = 0.0
        self.polyRight = []
        self.polyRight_px = []
        self.confRight = 0.0
        self.curvature = 0.0
        self.deviation = 0.0
        
    def calibrateCamera(self, pics="camera_cal/calibration*.jpg", nx=9, ny=6, debugFlag=False):
        """ Calibrate camera using pictures of chessboard
        Keyword arguments:
            pics -- pictures of chessboard with wildcard for glob
            nx -- number of horizontal internal corners
            ny -- number of vertical internal corners
            debugFlag -- Plot comparison between original and undistorted picture (only functional for udacity proj.)     
        """
        
        #Initialize empty arrays to store 3D and 2D points
        objpoints = []
        imgpoints = []
        
        # Sets up all the known 3D points in the chess board, always the same across images as the chess board does not change
        # Only the perspective we're looking at it
        objp = np.zeros((nx*ny,3),np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        
        #Read the calibration images
        images = glob.glob(pics)
        
        for fname in images:
            # Read the calibration image
            img = mpimg.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)       
            # If found, draw corners
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
            
        #Calibrate camera using all the pairs objpoints,imgpoints found before
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        #Store camera matrix and distortion coefficients in the LaneDetection object
        self.mtx = mtx
        self.dist = dist
        
        if debugFlag:
            # Compare original and undistorted image (on calibration1, most noticeable)
            img = mpimg.imread("camera_cal/calibration1.jpg")
            f, ((ax1, ax2)) = plt.subplots(1, 2)
            ax1.imshow(img)
            ax1.set_title('Original')
            ax2.imshow(cv2.undistort(img,mtx,dist,None,mtx))
            ax2.set_title('Undistorted')

    def undist(self, img):
        """Undistort image with parameters computed by calibrateCamera()"""
        return cv2.undistort(img,self.mtx,self.dist,None,self.mtx)


    def perpectiveTransform(self, img, debugFlag=False):
        """Perspective transform to view lane from above"""
        
        if debugFlag:
            # Compare original and undistorted image (on calibration1, most noticeable)
            img1 = mpimg.imread("camera_cal/calibration4.jpg")
            und = self.undist(img1)
            
            gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
            und = cv2.drawChessboardCorners(und, (8,6), corners, 1)
            
            src = np.float32([corners[0],corners[7],corners[40],corners[47]])
            dst = np.float32([[100,100],[1200,100],[100,900],[1200,900]])
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)

            f, ((ax1, ax2)) = plt.subplots(1, 2)
            ax1.imshow(img1)
            ax1.set_title('Original')
            ax2.imshow(warped)
            ax2.set_title('Transformed')
            
        return cv2.warpPerspective(img, self.Mperspective, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
            
    def setPerspective(self, src=[[280,674],[580,462],[708,462],[1038,674]], dst=[[350,720],[350,0],[930,0],[930,720]]):
        """Defines the perspective transform"""
        self.pSource = src
        self.pDest = dst
        self.Mperspective = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        
        
    def toSingleChannel(self, img):
        """Gets single channel image to analyse with the gradient"""
        
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0,255)):
    
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = self.toSingleChannel(img)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        # 6) Return this mask as your binary_output image
        thresh_min, thresh_max = thresh
        binarymask = np.zeros_like(scaled_sobel)
        binarymask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binarymask
    
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
    
        # Convert to grayscale
        gray = self.toSingleChannel(img)
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
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
        # Return the binary image
        return binary_output
    
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Convert to grayscale
        gray = self.toSingleChannel(img)
        # Take both Sobel x and y gradients
        abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Calculate the gradient magnitude
        gradmag = np.arctan2(abs_sobely, abs_sobelx)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    
        # Return the binary image
        return binary_output

    def colorThreshold(self, img, thresh=(90,255), debugFlag=False):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        L = hls[:,:,1]
        S = hls[:,:,2]
        mask = np.zeros_like(S)
        mask[(S > thresh[0]) & (S <= thresh[1]) & (L > 120)] = 1
        
        if debugFlag:
            f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10))
            ax1.imshow(img)
            ax1.set_title('Original')
            ax2.imshow(S, cmap="gray")
            ax2.set_title('S Channel')
            ax3.imshow(np.uint8(mask), cmap="gray")
            ax3.set_title('Mask')
            
        return mask

    def getLanes(self, image, debugFlag=False):
        """Combined Filter, color + gradients"""
        ksize = 7
        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 255))
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50, 255))
        #mag_binary = self.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255))
        #dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
        color_binary = self.colorThreshold(image, thresh=(30, 255))
        
        combined = np.zeros_like(gradx)
        combined[ ((gradx == 1) & (grady == 1)) | (color_binary == 1)] = 1

        if debugFlag:
            f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10,10))
            ax1.imshow(image)
            ax1.set_title('Original')
            ax2.imshow(combined, cmap="gray")
            ax2.set_title('Combined mask')
            
        return combined

    def drawFilled(self, img, vertices, color=[0,0,255] ):
        
        return cv2.fillConvexPoly(img, vertices, color)
        
    def draw_poly(self, img, vertices, color=[0, 0, 255], thickness=2 ):
        '''Draw ROI polgon'''
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(len(vertices[0])):
            x1, y1 = vertices[0][i-1]
            x2, y2 = vertices[0][i]
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        return line_img
    
        
    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to 
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).  
        
        Think about things like separating line segments by their 
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of 
        the lines and extrapolate to the top and bottom of the lane.
        
        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                
    def draw_frame(self, img, vertices, color=[255, 0, 0], thickness=2 ):
        """Draw border of polygon"""
        lines = []
        for i, p in enumerate(vertices):
            lines.append([(vertices[i-1][0], vertices[i-1][1], p[0], p[1])])

        self.draw_lines(img,lines, color=color, thickness=thickness )
        
    def sliding_window(self, binary_warped, margin=100, curv=0, debugFlag=False):
        """ Get polyfit for each lane
        Keyword arguments:
        margin -- Set the width of the windows +/- margin
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 10
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        conf_right = 0
        conf_left = 0
        # Step through the windows one by one
        for window in range(nwindows):
            
            marg = margin # int(margin * (1.0 + 0.3 * window/nwindows))
            #TODO: Higher curvatures may take advantage of apriori curv knowledge to adapt window position
            #TODO: Adapt margin with increasing distance from the camera, perspective transform blurs more far away from camera
            
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - marg
            win_xleft_high = leftx_current + marg
            win_xright_low = rightx_current - marg
            win_xright_high = rightx_current + marg
            if debugFlag:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
            conf_right += len(good_right_inds)
            conf_left += len(good_left_inds)
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        if len(leftx)==0 or len(lefty)==0:
            left_fit_m = self.polyLeft
            left_fit   = self.polyLeft_px
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fit_m = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        
        if len(rightx)==0 or len(righty)==0:
            right_fit_m = self.polyRight
            right_fit   = self.polyRight_px
        else:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fit_m = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        
        scene = np.zeros_like(out_img)
        scene[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        scene[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        if debugFlag:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)

        return (left_fit_m, right_fit_m, conf_left, conf_right, left_fit, right_fit, scene)
    
    def getCurvature(self, y_eval=710):
        """get curvature, average both lines"""
        return (self.calc_curvature(self.polyLeft, y_eval) + self.calc_curvature(self.polyRight, y_eval) )/2.0
    
    def getOffsetCenter(self):
        ploty = 719
        left_fitx  = self.polyLeft_px[0] *ploty**2 + self.polyLeft_px[1] *ploty + self.polyLeft_px[2]
        right_fitx = self.polyRight_px[0]*ploty**2 + self.polyRight_px[1]*ploty + self.polyRight_px[2]
        
        return ((left_fitx+right_fitx)/2.0 - 1280.0/2) * self.xm_per_pix
    
    def calc_curvature(self, poly, y_eval):
        """Computes curvature in meters"""    
        curverad = ((1 + (2*poly[0]*y_eval*self.ym_per_pix + poly[1])**2)**1.5) / np.absolute(2*poly[0])    
        return curverad
        
    def update(self, polyInfo):
        """Simple smoothing of the polynomial coefficients"""    
        iircoef = [0.95, 0.95, 0.95]
        
        (left_fit, right_fit, conf_left, conf_right, left_fit_px, right_fit_px) = polyInfo
        if len(self.polyLeft) == 0:
            self.polyLeft = left_fit
            self.polyLeft_px = left_fit_px
            
        if len(self.polyRight) == 0:
            self.polyRight = right_fit
            self.polyRight_px = right_fit_px
            
        for i in range(3):
            self.polyLeft[i] = left_fit[i]*(1.0-iircoef[i]) + iircoef[i]*self.polyLeft[i]
            self.polyRight[i] = right_fit[i]*(1.0-iircoef[i]) + iircoef[i]*self.polyRight[i] 
            self.polyLeft_px[i] = left_fit_px[i]*(1.0-iircoef[i]) + iircoef[i]*self.polyLeft_px[i]
            self.polyRight_px[i] = right_fit_px[i]*(1.0-iircoef[i]) + iircoef[i]*self.polyRight_px[i] 
        
    def lanePoly(self, top=0, bottom=719):
        
        ploty = np.linspace(top, bottom, 20 )
        left_fitx = self.polyLeft_px[0]*ploty**2 + self.polyLeft_px[1]*ploty + self.polyLeft_px[2]
        poly = np.stack((left_fitx,ploty),axis=1)
        ploty = top + bottom - ploty
        right_fitx = self.polyRight_px[0]*ploty**2 + self.polyRight_px[1]*ploty + self.polyRight_px[2]
        paux = np.stack((right_fitx,ploty),axis=1)
        poly = np.concatenate((poly, paux), axis=0)
        
        return poly
        
    def process_frame(self, frame):
        """ Complete frame processing.
        Updates laneDetection object with latest frame information
        Input:
            raw frame
        Output:
            frame with additional information     
        """
        #undistort image
        img0 = self.undist(frame)
        
        # Filter to get lane lines in the picture
        img1 = self.getLanes(img0)
        
        # Perspective transform the mask
        img2 = self.perpectiveTransform(img1)
        
        # Put it to binary range
        mask = np.zeros_like(img2)
        mask[ (img2 > 0) ] = 1
        
        # Get polynomials from binary mask
        (left_fit, right_fit, conf_left, conf_right,L,R, back) = self.sliding_window( np.uint8(mask))
        
        # Update Polynomial coefficient estimates
        self.update((left_fit, right_fit, conf_left, conf_right,L,R))
        
        #Add polygon to frame
        polVertices = np.int32(self.lanePoly())
        Pol = self.drawFilled( np.zeros_like(frame), polVertices, [0,255,0])
        
        res = cv2.resize(back+Pol,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
        sq = np.zeros_like(frame)
        xoff = 0
        yoff = sq.shape[1]-res.shape[1]
        sq[xoff:(xoff+res.shape[0]),yoff:(yoff+res.shape[1]),0:res.shape[2]] += res 
        
        
        # Get Curvature
        curv = int(self.getCurvature())
        off_ = self.getOffsetCenter()
        off  = int(np.abs(off_)*100)
        side = "left"
        if off_ < 0.0:
            side = "right"
        
        
        
        #Inverse perspective to get polygon back to the lane in original frame
        Minv = cv2.getPerspectiveTransform(np.float32(self.pDest), np.float32(self.pSource))
        iframe = cv2.warpPerspective(Pol+back, Minv, (Pol.shape[1],Pol.shape[0]), flags=cv2.INTER_LINEAR)
        
        iframe= weighted_img(frame, iframe, α=0.4, β=1.0, λ=0.0)
        
        iframe= weighted_img(iframe,sq, α=1.0, β=1.0, λ=0.0)
        #Text
        square = self.drawFilled( np.zeros_like(frame), np.int32([[0,0],[680,0],[680,120],[0,120]]),[255,255,255] )
        iframe= weighted_img(iframe, square, α=0.4, β=1.0, λ=0.0)
        iframe= cv2.putText(iframe, "Curv:"+ str(curv)  +"m", (0,50), cv2.FONT_HERSHEY_PLAIN, 3, [255,0,0],thickness=4)
        iframe= cv2.putText(iframe, "offset:"+ str(off)  +"cm to the "+ side, (0,100), cv2.FONT_HERSHEY_PLAIN, 3, [255,0,0],thickness=4)
        
        #Prepare output frame
        return iframe
        
if __name__ == "__main__":
    debugFlag = False
    LL = LaneDetection()
    
    # Calibrate camera based on chessboarde images
    LL.calibrateCamera()
    
    # Setup the perspective transform
    LL.setPerspective()
    
    if not debugFlag:
        white_output = 'white.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        print(clip1)
        white_clip = clip1.fl_image(LL.process_frame) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)

    if debugFlag:
        
        img = mpimg.imread("test_images/test6.jpg")
        # Process frame 
        img1 = LL.process_frame(img)
        f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10,10))
        ax1.imshow(img)
        ax1.set_title('Original')
        ax2.imshow(img1)
        ax2.set_title('Output')
    
    
    
    
    
    
    
