# encoding: utf-8
# Borrowed from https://github.com/L1aoXingyu/deploy-pytorch-model

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

# Initialize our Flask application and the PyTorch model.
IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250
IMAGE_CHANS = 3
IMAGE_DTYPE = "uint8"

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None

tags = ['ButtonCircle', 'ButtonSquare', 'Text', 'TextInput', 'ImageView', 'RadioButton', 'CheckBox']

model_path = "/Users/eash/Desktop/pytorch-faster-rcnn-master/vgg16_faster_rcnn_iter_60000.pth"

def base64_encode_image(a):
    return base64.b64encode(a.tobytes()).decode('utf-8')

def base64_decode_image(a, dtype):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

	# convert the string to a NumPy array using the supplied data
	# type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype)

    a = np.reshape(a, (500, 500, 3))
    #print(a)
    #a = cv2.resize(a, (500, 500))

    return a

def classify_process():

    load_model()
    print("Done LOADING!")

    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE-1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            #print(type(q["image"]))
            image = base64_decode_image(q["image"], IMAGE_DTYPE)
            
            if batch is None:
                batch = image
            else:
                batch.vstack([batch, image])

            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            results = [im_detect(model, image)]

            for (imageID, ans) in zip(imageIDs, results):
                #print(ans)
                output = []
                scores = ans[0]
                boxes = ans[1]
                print("Scores:", scores)
                print("Boxes: ", boxes)
                r = {"labels": scores.tolist(), "boxes": boxes.tolist()}
                output.append(r)

                db.set(imageID, json.dumps(output))
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        time.sleep(SERVER_SLEEP)

def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    
    model = vgg16()
    model.create_architecture(8,tag='default', anchor_scales=[8, 16, 32])
    #model.load_pretrained_cnn('./webapp/var/www/models/vgg16_faster_rcnn_iter_60000.pkl')
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    if not torch.cuda.is_available():
        model._device = 'cpu'
    model.to(model._device)

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
            #print(image.dtype)
            image = cv2.resize(image, (500, 500))
            #print(image.shape)
            image = image.copy(order='C')

            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            # keep looping until our model server returns the output
			# predictions
            while True:
                output = db.get(k)

                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                
                time.sleep(CLIENT_SLEEP)

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':

    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    os.system("redis-cli flushall")

    app.run(port=4000)
