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

#from app import init_paths, load_model, predict_result
import json
from lib.model.config import cfg
from lib.model.test import im_detect
from lib.model.nms_wrapper import nms

from lib.utils.timer import Timer
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from io import BytesIO
#import PyQt5

#matplotlib.use('qt5agg')
import numpy as np
import os, cv2
import argparse

from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1

import torch
import matplotlib.pyplot as plt

PLOT_FOLDER = './var/www/plots/'

CLASSES = ('__background__',  # always index 0
                     'ButtonCircle', 'ButtonSquare', 'CheckBox', 'ImageView', 'RadioButton', 'TextInput', 'Text')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

fig = None
ax = None
data = None

def create_plot(scores, boxes, im_file):
    """Detect object classes in an image using pre-computed object proposals."""
    global fig 
    global ax
    global data
    fig, ax = plt.subplots()

    data = {'results':list()}
    # Load the demo image
    im = cv2.imread(im_file)
    assert(im is not None)

    data = {'results':list()}

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, cls_ind,dets, thresh=CONF_THRESH)

    j = plt.gcf()

    with open('var/www/downloads/json/data.json', 'w') as outfile:  
        json.dump(data, outfile, indent=4)

    #j.savefig('./static/plot.png')
    png_output = BytesIO()
    j.savefig(png_output, format='png')
    png_output.seek(0)
    import base64
    png_output = base64.b64encode(png_output.getvalue())
    return png_output

def vis_detections(im, class_name, cls_ind, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    global fig 
    global ax
    global data

    data['results'].append({class_name:list()})
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots()
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=6, color='white')
        
        data['results'][cls_ind-1][class_name].append({
            'score': str(score),
            'bl_point':(str(bbox[0]), str(bbox[1])),
            'width':str(bbox[2] - bbox[0]),
            'height':str(bbox[3] - bbox[1])
        })
    plt.axis('off')
    plt.tight_layout()
    plt.draw()