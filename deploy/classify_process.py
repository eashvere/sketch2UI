# Borrowed from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/


import os
import io
import json
import numpy as np
import flask
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50
import cv2
import redis
import base64
import uuid
import time
from threading import Thread
import sys

from lib.nets.vgg16 import vgg16
from lib.model.test import im_detect
from helpers import base64_decode_image, base64_encode_image
import settings

db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

tags = ['ButtonCircle', 'ButtonSquare', 'Text', 'TextInput', 'ImageView', 'RadioButton', 'CheckBox']

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/vgg16_faster_rcnn_iter_60000.pth")

def start_classify():

    model = load_model()
    print("Done LOADING!")

    while True:
        queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE-1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            #print(type(q["image"]))
            image = base64_decode_image(q["image"], settings.IMAGE_DTYPE)
            
            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])

            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            print("* Batch size: {}".format(batch.shape))
            results = [im_detect(model, image)]

            for (imageID, ans) in zip(imageIDs, results):
                #print(ans)
                output = []
                scores = ans[0]
                boxes = ans[1]
                r = {"labels": scores.tolist(), "boxes": boxes.tolist()}
                output.append(r)

                db.set(imageID, json.dumps(output))
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

        time.sleep(settings.SERVER_SLEEP)

def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    
    model = vgg16()
    model.create_architecture(8,tag='default', anchor_scales=[8, 16, 32])
    #model.load_pretrained_cnn('./webapp/var/www/models/vgg16_faster_rcnn_iter_60000.pkl')
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    if not torch.cuda.is_available():
        model._device = 'cpu'
    model.to(model._device)
    return model

if __name__ == '__main__':
    start_classify()