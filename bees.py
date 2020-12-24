import cv2 as cv
import numpy as np
import imutils
import time
from beepipe import BeePipeline
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

from moviepy.editor import ImageSequenceClip

#Set these values according to each video
#Video must be 1200x1280 dimension
videoPath = ''
fps = 15

#Bring in computer vision pipeline
pipe = BeePipeline()

x = []
y = []
frame_count = 0

#The video that the code will parse through
cap = cv.VideoCapture(videoPath)

#Save the first frame, which just shows the background, so that we can compare it to future frames
ret, frame1 = cap.read()
pipe.process(frame1)
frame1_rgb = pipe.rgb_threshold_output

#Set up the graph
fig = plt.figure()
plot = fig.add_subplot(411)
plot.autoscale(enable=True, axis='both', tight=None)

#We'll save processed video frames here
frame_array = []

#While there are more video frames to process
while cap.isOpened():

    frame_count += 1

    #Run the new frame through the pipeline
    ret, frame = cap.read()
    unprocessed = frame
    pipe.process(frame)
    frame_rgb = pipe.rgb_threshold_output

    #Find the difference between this new frame and our first frame (the background)
    diff = cv.absdiff(frame_rgb, frame1_rgb)
    thresh = cv.threshold(diff, 200, 255, cv.THRESH_BINARY)[1]

    #Filter out the differences that couldn't be bees
    pipe.process_diff(thresh)
    thresh_contours = pipe.filter_contours_output
    cv.drawContours(unprocessed, thresh_contours, -1, (255, 255, 255), 2)

    #Put the number of bees detected on the frame
    cv.putText(unprocessed, str(len(pipe.filter_contours_output)), (50,100), cv.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    #Plot the new point, where x = frame number and y = number of bees
    x.append(frame_count)
    y.append(str(len(pipe.filter_contours_output)))
    plot.plot(x, y, color='red', linewidth=1)
    plt.draw()
    plt.pause(.001)

    #Format the graph and show it under the video frames
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    two_images = np.vstack((unprocessed, data))
    cropped_image = two_images[0:1200, 0:1280]
    cv.imshow('Bee Counter', cropped_image)

    #Save the shown frames
    cropped_image_bgr = cv.cvtColor(cropped_image, cv.COLOR_RGB2BGR)
    frame_array.append(cropped_image_bgr)

    if cv.waitKey(1) == ord('q'):
         break

cap.release()
cv.destroyAllWindows()

#Create and save the new, processed video
pathOut = videoPath + '_count.mp4'
clip = ImageSequenceClip(frame_array, fps=fps)
clip.write_videofile(pathOut, fps=fps)
