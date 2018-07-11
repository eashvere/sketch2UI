import json
import torch
from PIL import Image
import os
import io
import xml.etree.cElementTree as ET
from flask import render_template, request, Flask, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import requests

UPLOAD_FOLDER = 'var/www/uploads'
EXTENSIONS = set(['jpg', 'png', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PyTorch_REST_API_URL = 'http://127.0.0.1:4000/predict'

def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    assert(image is not None)
    payload = {'image': image}
    #print(payload)

    # Submit the request.
    #print(requests.post(PyTorch_REST_API_URL, files=payload)
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        return r['predictions'][0]['labels'], r['predictions'][0]['labels']
    # Otherwise, the request failed.
    else:
        print('Request failed')

def file_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS

@app.route("/")
def home_screen():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            #flash('No File in the form subimitted')
            return redirect('/')
        file = request.files['file']
        if file.filename == '':
            flash('No File submitted')
            return redirect('/')
        file.filename = 'image.jpg'
        if file and file_allowed(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(file_path)
            label, box = predict_result(file_path)
            return render_template('result.html', label=label, box=box)


    return redirect('/')



if __name__ == '__main__':
    app.secret_key = 'lolUWiscool'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True, port=5000)