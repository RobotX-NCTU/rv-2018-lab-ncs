#!/usr/bin/env python
import numpy as np
import cv2 
import sys
from sensor_msgs.msg import CompressedImage, Image
import rospy
from cv_bridge import CvBridge, CvBridgeError
from placard_msgs.msg import RegionProposalImage

class MserRegionExtraction(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))

        #Variables
        self.bridge = CvBridge()
        self.counts = 0

        #Publisher
        self.pub_image_region_mser = rospy.Publisher("~region_proposal", RegionProposalImage, queue_size=1)
        self.pub_image_all_mser = rospy.Publisher("~image/image_mser_all_region", Image, queue_size=1)
        self.pub_image_origin   = rospy.Publisher("~image/image_raw", Image, queue_size=1)

        #Subscriber
        self.sub_image = rospy.Subscriber("~compressed/image_compressed", CompressedImage, self.cbCompressedImage, queue_size=1)

    def cbCompressedImage(self, image_msg):
        if self.counts == 15:
            
            self.counts = 0
            # ***************************************************************
            # Transfer compressed image to opencv image
            # Pi camera image shape (480, 680, 3)
            # ***************************************************************
            if len(image_msg.data) is 0 :
                return
            np_arr = np.fromstring(image_msg.data, np.uint8)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            img_msg = Image()
            img_msg.header.stamp = rospy.Time.now()
            img_msg = self.bridge.cv2_to_imgmsg(img_cv, "bgr8")
            self.pub_image_origin.publish(img_msg)
    
            # ***************************************************************
            # Transfer BGR to gray image
            # ***************************************************************
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # ***************************************************************
            # Create MSER and get the region proposal
            # ***************************************************************
            mser = cv2.MSER_create()
            imgContours = img_cv.copy()
            regions, _ = mser.detectRegions(img_gray)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

            # ***************************************************************
            # Extract each rectangle of region proposal and publish
            # ***************************************************************
            counts = 0
            for i, contour in enumerate(hulls):
                x,y,w,h = cv2.boundingRect(contour)
                img_region = img_cv[y:y+h, x:x+w]           
                if w < 1.4*h and h < 1.4*w and h*w < 10000:
                    counts += 1
                    img_msg = Image()
                    img_msg.header.stamp = rospy.Time.now()
                    img_msg = self.bridge.cv2_to_imgmsg(img_region, "bgr8")
            
                    region_msg = RegionProposalImage()
                    region_msg.image_region = img_msg
                    region_msg.x = x
                    region_msg.y = y
                    region_msg.width = w
                    region_msg.height = h
                    self.pub_image_region_mser.publish(region_msg)
                    #Draw mser rectangle
                    cv2.rectangle(imgContours,(x, y),(x+w, y+h),(0,255,0),3)
            #print "number of region potential = ", counts

            # ***************************************************************
            # Visualization of region proposal
            # ***************************************************************
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


