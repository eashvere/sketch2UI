import json
import torch
from PIL import Image
import os
import io
import xml.etree.cElementTree as ET
from flask import render_template, request, Flask, redirect, url_for, flash, send_from_directory
from simple_request import predict_result
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'var/www/uploads'
EXTENSIONS = set(['jpg', 'png', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#def create_xml(labels, box, flipped)

def file_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS

@app.route("/")
def home_screen():
    return render_template('home.html')

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
            label, box, flipped = predict_result(Image.open(file_path))


    return redirect('/')



if __name__ == '__main__':
    app.secret_key = 'lolUWiscool'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True)