# encoding: utf-8
# Borrowed from https://github.com/L1aoXingyu/deploy-pytorch-model

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

from lib.nets.vgg16 import vgg16
from lib.model.test import im_detect

model_path = ""

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = True

tags = ['ButtonCircle', 'ButtonSquare', 'Text', 'TextInput', 'ImageView', 'RadioButton', 'CheckBox']

model_path = "./deploy/models/vgg16_faster_rcnn_iter_60000.pth"

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

def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """

    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    #image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image, volatile=True)


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
            

            # Preprocess the image and prepare it for classification.
            #image = prepare_image(image, target_size=(500, 500))

            # Classify the input image and then initialize the list of predictions to return to the client.
            scores, boxes = im_detect(model, image)

            data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            r = {"labels":scores.tolist(), "boxes": boxes.tolist()}
            data['predictions'].append(r)

            # Indicate that the request was a success.
            data["success"] = True
            #print(data)

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run(port=4000)
