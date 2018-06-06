#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import time
import math
import sys
from cv_bridge import CvBridge, CvBridgeError
from mvnc import mvncapi as mvnc
import cv2
import numpy as np

class PlacardNcsPredictionNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))

        #Variables
        self.bridge = CvBridge()
        self.model_name = rospy.get_param('~caffe_model')
        self.input_shape = rospy.get_param('~input_shape') 
        rospy.loginfo('[%s] caffe model name = %s' %(self.node_name, self.model_name))
        rospy.loginfo('[%s] input x`shape name = %s' %(self.node_name, self.input_shape))
        self.model_Base_Dir = '../models/' + self.model_name + '/'
        
        #Set up NCS
        #self.initial()

        #Publisher

        #Subscriber
        self.sub_image_region = rospy.Subscriber("~image/image_mser_region", Image, self.cbImage, queue_size=1)

    def cbImage(self, image_msg):

        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
        img = cv2.resize(img, self.dim)
        img = img.astype(np.float32)
        
        img_m = np.zeros((self.dim[0], self.dim[1], 1), np.float32)
        img_m[:] = (128.0)
        img = cv2.subtract(img, img_m)	
        img = img * 0.0078125
        
        # Send the image to the NCS
        self.graph.LoadTensor(img.astype(np.float16), 'user object')

        output, userobj = self.graph.GetResult()

        order = output.argsort()[::-1][:4]
        print('\n------- predictions --------')
        for i in range(0, 3):
            print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]*100) + '%) is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) )        

    def initial(self):

        self.device_work = False
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        self.deviceCheck()
        shape_txt = ""
        shape = []
        with open(self.model_Base_Dir + self.input_shape + '.prototxt', 'r') as file:
            shape_txt = file.read().replace('\n', ' ')
        for s in shape_txt.split():
            if s.isdigit():
                shape.append(int(s))
        
        self.dim = (shape[2], shape[3])

    def initialDevice(self):
        # set the blob, label and graph
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
        #check device usable
        self.devices = mvnc.EnumerateDevices()
        if len(self.devices) == 0:
            self.device_work = False
            rospy.loginfo('[%s] NCS device not found' %(self.node_name))
            
        else:
            self.device_work = True
            rospy.loginfo('[%s] NCS device found' %(self.node_name))
            self.initialDevice()

    def onShutdown(self):
        #close divice when shutdown
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
