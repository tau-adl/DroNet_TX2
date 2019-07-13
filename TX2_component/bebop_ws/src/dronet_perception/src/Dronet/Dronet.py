#!/usr/bin/env python
import rospy
from dronet_perception.msg import CNN_out
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty
import utils
import time
import numpy as np

from keras import backend as K

TEST_PHASE=0

class Dronet(object):
    def __init__(self,
                 json_model_path,
                 weights_path, target_size=(200, 200),
                 crop_size=(150, 150),
                 imgs_rootpath="../models"):

        self.pub = rospy.Publisher("cnn_predictions", CNN_out, queue_size=5)
        self.feedthrough_sub = rospy.Subscriber("state_change", Bool, self.callback_feedthrough, queue_size=1)
        self.land_sub = rospy.Subscriber("land", Empty, self.callback_land, queue_size=1)
        
        self.img_sub = rospy.Subscriber("camera", Image, self.callback_image, queue_size=1)
        self.rdy = False
        self.data = None

        self.use_network_out = False
        self.imgs_rootpath = imgs_rootpath

        # Set keras utils
        K.set_learning_phase(TEST_PHASE)

        # Load json and create model
        model = utils.jsonToModel(json_model_path)
        # Load weights
        model.load_weights(weights_path)
        print("Loaded model from {}".format(weights_path))

        model.compile(loss='mse', optimizer='sgd')
        self.model = model
        self.target_size = target_size
        self.crop_size = crop_size

    def callback_feedthrough(self, data):
        self.use_network_out = data.data

    def callback_land(self, data):
        self.use_network_out = False
        
    def callback_image(self, data):
        self.data = data
        self.rdy = True

    def run(self):
        t = time.time()
        while not rospy.is_shutdown():
            msg = CNN_out()
            msg.header.stamp = rospy.Time.now()
            data = None
            rate = rospy.Rate(60)
            
            while self.rdy is False:
                rate.sleep()
            self.rdy = False
            
            if self.use_network_out:
                print("Publishing commands!")
            #else:
            #    print("NOT Publishing commands!")


            cv_image = utils.callback_img(self.data, self.target_size, self.crop_size,
                self.imgs_rootpath, self.use_network_out)
            #print("###\n###\n")
            #print("$%6.3f, $%6.3f, $%6.3f, $%6.3f" % (cv_image[0,0,0], cv_image[0,20,0], cv_image[0,30,0], cv_image[0,50,0]))
            #print("###\n###\n")
            outs = self.model.predict_on_batch(cv_image[None])
            steer, coll = outs[0][0], outs[1][0]

            print("{0}: got commands steer={1}, coll={2}".format(time.time()-t, outs[0][0], outs[1][0]))
            #toc = time.time()
            #if ( (toc - t) >= 1 ):
            #    t = toc
            #    print("1 sec mark")
            msg.steering_angle = steer
            msg.collision_prob = coll
            self.pub.publish(msg)
