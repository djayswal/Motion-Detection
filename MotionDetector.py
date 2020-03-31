import time
import datetime
import cv2
import imutils


def motiondetection():
  videocapture = cv2.VideoCapture(0)#[0] #Value [0] selects the device's default camera
  time.sleep(2)

  firstframe = None #Initiates First frame
  while True:
    frame1 = videocapture.read()[1] #Take the 1st frame
    text = "Empty" # Shows that 1st frame unoccupied

    greyscale_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) # Here we took the frame and make it grey scale

    gaussian_frame = cv2.GaussianBlur(greyscale_frame, (101,101),0) # We took grey scale and use gaussian blur on it
    # (21,21) is the image in the kernel. Its odd because center of the image will be invalid if we take even

    blur_frame = cv2.blur(gaussian_frame,(51,51))
    # (5,5) this is also a image kernel grid which will go from left to right and all the way to down the image and it will blur the pixels using the middle value.

    greyscale_image = blur_frame

    if firstframe is None:
      firstframe = greyscale_image
    else:
      pass # It captures the diference between first and the next frame

    frame1 = imutils.resize(frame1, width= 500)
    frame_delta = cv2.absdiff(firstframe, greyscale_image)
    # Calculates the diff between each element between two images

    thresh = cv2.threshold(frame_delta, 100, 255, cv2.THRESH_BINARY)[1]
    #threshold gives two output retval and threshold image, using [1] on the end iam selecting the threshold image that is produced

    dilate_image = cv2.dilate(thresh, None,iterations = 2) # dilate means dilate, grow,expand
    # the effect on a binary image is to enlarge the white pixels in the foreground which are white

    cnt = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # Contours gives 3 different outputs-> image, Contour, Hierarchy

    for c in cnt:
      if cv2.contourArea(c) > 500: # It will be white if the area is greater than 500
        (x, y, w, h) = cv2.boundingRect(c)

        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0),2)

        text = "Occupied"
      else:
        pass
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame1, datetime.datetime.now().strftime('%A %d %B%Y %I:%M:%S%p'),(10,frame1.shape[0]-10), font, 0.50, (0,0,255),1)
    cv2.putText(frame1, f'Room Status: {text}',(10,30), font, 0.7,(0,0,255),2) #Shows the text 

    cv2.imshow('Security Camera',frame1)  #Shows the main camera
    cv2.imshow('Threshold(Foreground Mask) B&W',dilate_image) # Shows the Black and white colours
    #it will show black if its vacant/Empty and white where camera finds any moving object
    cv2.imshow('Frame_delta(blured greyscale)',frame_delta) # Shows the grey scale blurred image

    if cv2.waitKey(1) & 0xFF == ord('q'): # Here 0xFF is the hexadecimal number which is all 8 ones(1)
      cv2.destroyAllWindows()
      break

if __name__ == '__main__':
  motiondetection()      
