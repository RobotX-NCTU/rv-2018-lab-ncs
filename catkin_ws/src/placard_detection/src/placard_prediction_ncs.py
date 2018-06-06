#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage
from placard_msgs.msg import RegionProposalImage
import os
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from mvnc import mvncapi as mvnc
import cv2
import numpy as np

class PlacardNcsPredictionNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))
        # ***************************************************************
        # Get the position of graph folder
        # ***************************************************************
        self.model_name = rospy.get_param('~model_name')
        rospy.loginfo('[%s] model name = %s' %(self.node_name, self.model_name))
        rospack = rospkg.RosPack()
        self.model_Base_Dir = rospack.get_path('placard_detection') + '/models/' + self.model_name + '/'

        # ***************************************************************
        # Variables
        # ***************************************************************
        self.bridge = CvBridge()
        self.pred_count = 0
        self.dim = (227, 227)
        self.img = None;

        # ***************************************************************
        # Set up NCS
        # ***************************************************************
        self.initial()

        #Publisher
        self.pub_image_pred = rospy.Publisher("~image_pred", Image, queue_size=1)

        #Subscriber
        self.sub_image_region = rospy.Subscriber("~region_proposal", RegionProposalImage, self.cbRegion, queue_size=1)
        self.sub_image_raw = rospy.Subscriber("~image/image_raw", Image, self.cbImageRaw, queue_size=1)

    def cbRegion(self, region_msg):
        # ***************************************************************
        # Make sure that the device is working
        # ***************************************************************
        
        if self.device_work is True:
            
            # ***************************************************************
            # Preprocessing
            # ***************************************************************
            if(region_msg.image_region.width is 0):
                return
            img = self.bridge.imgmsg_to_cv2(region_msg.image_region, "bgr8")
            img = cv2.resize(img, self.dim)
            img = img.astype(np.float32)
            img_m = np.zeros((self.dim[0], self.dim[1], 3), np.float32)
            img_m[:] = (104, 116, 122)
            img = cv2.subtract(img, img_m)
            
            # ***************************************************************
            # Send the image to NCS and get the result
            # ***************************************************************
            self.graph.LoadTensor(img.astype(np.float16), 'user object')
            output, userobj = self.graph.GetResult()
            order = output.argsort()[::-1][:4]

            # ***************************************************************
            # Probability thresold
            # ***************************************************************
            if(output[order[0]]*100 >= 95):
                #cv2.imwrite("/home/tony/ncs/ncsdk/examples/caffe/CaffeNet/image/" + str(self.img_count) + ".jpg", img_cv)
                print('\n-------region predictions --------')
                x = region_msg.x
                y = region_msg.y
                w = region_msg.width
                h = region_msg.height
                if str(order[0]) is str(0):
                    cv2.rectangle(self.img,(x, y),(x+w, y+h),(255,0,0),3)
                    cv2.putText(self.img,'blue_circle',(x-50,y-5),cv2.FONT_HERSHEY_COMPLEX,0.3,(255,0,0),1)
                elif str(order[0]) is str(1):
                    cv2.rectangle(self.img,(x, y),(x+w, y+h),(0,255,0),3)
                    cv2.putText(self.img,'green_triangle',(x-50,y-5),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,255,0),1)
                elif str(order[0]) is str(2):
                    cv2.rectangle(self.img,(x, y),(x+w, y+h),(0,0,255),3)
                    cv2.putText(self.img,'red_cross',(x-50,y-5),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,255),1)
                elif str(order[0]) is str(3):
                    cv2.rectangle(self.img,(x, y),(x+w, y+h),(0,0,0),3)
                    cv2.putText(self.img,'background',(x-50,y-5),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0),1)
                img_msg = Image()
                img_msg = self.bridge.cv2_to_imgmsg(self.img, "bgr8")
                self.pub_image_pred.publish(img_msg)

                for i in range(0, 4):
                    print str(self.pred_count), (' prediction ' + str(i) + ' (probability ' + str(output[order[i]]*100) + '%) is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) )        
                self.pred_count += 1

    def cbImageRaw(self, img_msg):
        self.img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
    def initial(self):
        self.device_work = False
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        self.deviceCheck()

    def initialDevice(self):
        # ***************************************************************
        # Load the graph and label file
        # ***************************************************************
        self.device = mvnc.Device(self.devices[0])
        labels_file=self.model_Base_Dir + 'label.txt'
        self.labels = np.loadtxt(labels_file,str,delimiter='\t')
        self.device.OpenDevice()

        network_blob=self.model_Base_Dir + 'graph'

        #Load blob
        with open(network_blob, mode='rb') as f:
            blob = f.read()

        self.graph = self.device.AllocateGraph(blob)
        
    def deviceCheck(self):
        # ***************************************************************
        # Check device found
        # ***************************************************************

        self.devices = mvnc.EnumerateDevices()
        if len(self.devices) == 0:
            self.device_work = False
            rospy.loginfo('[%s] NCS device not found' %(self.node_name))
            
        else:
            self.device_work = True
            rospy.loginfo('[%s] NCS device found' %(self.node_name))
            self.initialDevice()

    def onShutdown(self):
        # ***************************************************************
        # Close device
        # ***************************************************************
        if(self.device_work==True):
            self.device_work=False
            rospy.sleep(0.5)
            self.graph.DeallocateGraph()
            self.device.CloseDevice()

if __name__ == '__main__':
        rospy.init_node('placard_ncs_prediction_node',anonymous=False)
        placard_ncs_prediction_node = PlacardNcsPredictionNode()
        rospy.on_shutdown(placard_ncs_prediction_node.onShutdown)
        rospy.spin() 
