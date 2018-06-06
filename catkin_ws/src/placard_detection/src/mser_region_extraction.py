#!/usr/bin/env python
import numpy as np
import cv2 
import sys
from sensor_msgs.msg import CompressedImage, Image
import rospy
from cv_bridge import CvBridge, CvBridgeError

class MserRegionExtraction(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))

        #Variables
        self.bridge = CvBridge()
        self.counts = 0

        #Publisher
        self.pub_image_region_mser = rospy.Publisher("~image/image_mser_region", Image, queue_size=1)
        self.pub_image_all_mser = rospy.Publisher("~image/image_mser_all_region", Image, queue_size=1)

        #Subscriber
        self.sub_image = rospy.Subscriber("~compressed/image_compressed", CompressedImage, self.cbCompressedImage, queue_size=1)

    def cbCompressedImage(self, image_msg):
        if self.counts == 3:
            self.counts = 0

            if len(image_msg.data) is 0 :
                return
            np_arr = np.fromstring(image_msg.data, np.uint8)

            #image shape (480, 680, 3)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


            #Transfer BGR to Gray image
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            #Create MSER and detect regions
            mser = cv2.MSER_create()
            imgContours = img_cv.copy()
            regions, _ = mser.detectRegions(img_gray)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

            #Extract each rectangle region and publish
            counts = 0
            for i, contour in enumerate(hulls):
                x,y,w,h = cv2.boundingRect(contour)
                img_region = img_cv[y:y+h, x:x+w]
                img_size = w*h             

                #if blue > thresold or red>thresold or green > thresold:
                if w < 1.3*h and h < 1.3*w and h*w < 10000:
                    counts += 1
                    img_msg = Image()
                    img_msg.header = rospy.Time.now
                    img_msg = self.bridge.cv2_to_imgmsg(img_region, "bgr8")
                    self.pub_image_region_mser.publish(img_msg)
                    #Draw mser rectangle
                    cv2.rectangle(imgContours,(x, y),(x+w, y+h),(0,255,0),3)
            print "number of region potential = ", counts

            image_all_mser_image = Image()
            image_all_mser_image.header = rospy.Time.now
            image_all_mser_image = self.bridge.cv2_to_imgmsg(imgContours, "bgr8")
            
            self.pub_image_all_mser.publish(image_all_mser_image)

        self.counts += 1

    def onShutdown(self):
        rospy.loginfo("[%s] Shutdown." %(self.node_name))

if __name__ == '__main__':
        rospy.init_node('mser_region_extract_node',anonymous=False)
        mser_region_extract_node = MserRegionExtraction()
        rospy.on_shutdown(mser_region_extract_node.onShutdown)
        rospy.spin() 


