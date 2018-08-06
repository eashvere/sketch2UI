import json
import torch
from PIL import Image
import os
import io
import xml.etree.cElementTree as ET
from flask import render_template, request, Flask, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import requests
import numpy as np
from visual import create_plot

UPLOAD_FOLDER = 'var/www/uploads'
EXTENSIONS = set(['jpg', 'png', 'jpeg'])
png_output = None


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENV'] = 'development'

PyTorch_REST_API_URL = 'http://127.0.0.1:4000/predict'

extra_dirs = ['templates/', 'static/']
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in os.walk(extra_dir):
        for filename in files:
            filename = os.path.join(dirname, filename)
            if os.path.isfile(filename):
                extra_files.append(filename)

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
        return r['predictions'][0]['labels'], r['predictions'][0]['boxes']
    # Otherwise, the request failed.
    else:
        print('Request failed')

def file_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS

@app.route("/")
def home_screen():
    return render_template('index.html')

@app.route("/home_page")
def redirect_back_home():
    return redirect(url_for('home_screen'))

@app.route("/oops")
def oops():
    return "OOPS! You didn't input any thing, please try again"

@app.route("/instruct")
def instructions():
    return render_template('instruct.html')

@app.route("/pubs")
def pubs():
    return render_template('pubs.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #error = None
    global png_output
    if request.method == 'POST':
        if 'file' not in request.files:
            #flash('No File in the form subimitted')
            return redirect('/oops')
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
            label = np.array(label)
            box = np.array(box)
            png_output = create_plot(label, box, os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template('result.html', image_data=png_output.decode('utf-8'))

app.route("/download")
def download(files):
    pass
    #TODO Create this download endpoint!




if __name__ == '__main__':
    app.secret_key = 'lolUWiscool'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True, port=5000)