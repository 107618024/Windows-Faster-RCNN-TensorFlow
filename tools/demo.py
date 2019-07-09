#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from natsort import natsorted
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'perosn')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_5000.ckpt',),'res50': ('res50_faster_rcnn_iter_5000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name,image_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
   # if len(inds) == 0:
    #   return
    name=image_name.split(".")[0]

#    path_txt=os.path.join("C:/Users/gary/Desktop/show",name+".txt")
#   f=open(path_txt, mode='a') 
    path=img_name.split("_")
    data=path[2].split(".")
#    im = im[:, :, (2, 1, 0)]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        w=bbox[2]-bbox[0]
        h=bbox[3]-bbox[1]
        f.write(str(int(data[0])+1)+","+str(bbox[0])+","+str(bbox[1])+","+str(w)+","+str(h)+","+str(score)+"\n")
#        f.write(image_name+" "+str(class_name)+" "+str(bbox[0])+" "+str(bbox[1])+" "+str(w)+" "+str(h)+" "+str(score)+"\n")
#        f.write(str(int(class_name)-1)+" "+str(score)+" "+str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+"\n")
        cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0),4)
#        if class_name==1:
#            cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0),4)
#        if class_name==2:
#            cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255),4)
#        if class_name==3:
#            cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0),4)  
            
        cv2.putText(im, str(str(class_name)+format(score,' .3f')), (bbox[0],bbox[1]), 1, 1, (0, 0, 255), 2)
        cv2.imwrite('C:/Users/gary/Desktop/show/{}'.format(image_name),im)  
#    f.close()

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls,image_name, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    demonet = 'res50'
    dataset = 'pascal_voc'
    tfmodel = os.path.join(r'C:\Users\gary\Desktop\tf-faster-rcnn-master\tf-faster-rcnn-master\output', demonet, DATASETS[dataset][0], 'Train',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res50':
        net = resnetv1(num_layers=50)
    else:
        raise NotImplementedError
    n_classes = len(CLASSES)     
    net.create_architecture("TEST", n_classes,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    img_paths = r"C:\Users\gary\Desktop\output\set06",r"C:\Users\gary\Desktop\output\set07",r"C:\Users\gary\Desktop\output\set08",r"C:\Users\gary\Desktop\output\set09",r"C:\Users\gary\Desktop\output\set10"
    out_path = r"C:\Users\gary\Desktop\code3.2.1\data-USA\res\B09"
    for img_path in img_paths:
        for img_name in natsorted(os.listdir(img_path)):
                 # if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                 #  continue
                 path=img_name.split("_")
                 path_txt=os.path.join(out_path,path[0],path[1]+".txt")
                 f=open(path_txt, mode='a')
                 print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                 print('Demo for data/demo/{}'.format(img_name))
                 demo(sess, net, img_name)
                 f.close    
