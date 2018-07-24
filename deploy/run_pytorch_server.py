# encoding: utf-8
# Borrowed from https://github.com/L1aoXingyu/deploy-pytorch-model
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

# Initialize our Flask application and the PyTorch model.

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
 
	# resize the input image and preprocess it
	image = image.resize(target)
	image = np.array(image)
	#image = np.expand_dims(image, axis=0)
	#image = imagenet_utils.preprocess_input(image)
 
	# return the processed image
	return image

@app.route("/")
def homepage():
    return "Hello! This is the homepage of the Sketch2UI API \n Use this link with the /predict at the end to access this API"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
 
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# classification
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image,
				(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
 
			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = np.copy(image, order="C")
 
			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = base64_encode_image(image)
			d = {"id": k, "image": image}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
 
			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)
 
				# check to see if our model has classified the input
				# image
				if output is not None:
					# add the output predictions to our data
					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)
 
					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break
 
				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(settings.CLIENT_SLEEP)
 
			# indicate that the request was a success
			data["success"] = True
 
	# return the data dictionary as a JSON response
	return flask.jsonify(data)


if __name__ == '__main__':

    print("* Starting model service...")
    os.system("redis-cli flushall")

    app.run(port=4000)
