import json
import torch
from PIL import Image
import os
import io
import xml.etree.cElementTree as ET
from flask import render_template, request, Flask, redirect, url_for, flash, send_file, send_from_directory
from werkzeug.utils import secure_filename
import requests
import numpy as np
from visual import create_plot
import zipfile
import shutil

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

def delete_files():
    shutil.rmtree("var/www/downloads/json")
    os.mkdir("var/www/downloads/json")
    shutil.rmtree("var/www/downloads/xml")
    os.mkdir("var/www/downloads/xml")
    shutil.rmtree("var/www/downloads/zip")
    os.mkdir("var/www/downloads/zip")
    shutil.rmtree("var/www/uploads")
    os.mkdir("var/www/uploads")

def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    assert(image is not None)
    payload = {'image': image}
    #print(payload)

    # Submit the request.
    try:
        r = requests.post(PyTorch_REST_API_URL, files=payload).json()
    except Exception as e:
        print(e)

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

@app.errorhandler(403)
def page_forbidden(e):
    return render_template('errors/403.html'), 403

@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('errors/500.html'), 500

@app.route("/instruct")
def instructions():
    return render_template('instruct.html')

@app.route("/pubs")
def pubs():
    return render_template('pubs.html')

@app.route("/about")
def about():
    return render_template('about.html')

app.route("/download_json")
def jsons():
    zipf = zipfile.ZipFile('json.zip', mode='w', compression=zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk('/var/www/downloads/json/'):
        for file in files:
            print(file)
            zipf.write('/var/www/downloads/json/'+file)
    zipf.close()
    try:
        return send_from_directory('/', 'json.zip', as_attachment=True)
    except Exception as e:
        return str(e)

app.route("/download_xml")
def xmls():
    zipf = zipfile.ZipFile('/var/www/downloads/zip/xml.zip', mode='w', compression=zipfile.ZIP_DEFLATED)
    for root,dirs, files in os.walk('/var/www/downloads/xml/'):
        for file in files:
            zipf.write('/var/www/downloads/xml/'+file)
    zipf.close()
    try:
        return send_from_directory('/var/www/downloads/zip', 'xml.zip', as_attachment=True)
    except Exception as e:
        return str(e)

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
            delete_files()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(file_path)
            label, box = predict_result(file_path)
            label = np.array(label)
            box = np.array(box)
            png_output = create_plot(label, box, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return render_template('result.html', image_data=png_output.decode('utf-8'))
            
    return render_template('result.html', image_data=png_output.decode('utf-8'))




if __name__ == '__main__':
    app.secret_key = 'lolUWiscool'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.register_error_handler(403, page_forbidden)
    app.register_error_handler(404, page_not_found)
    app.register_error_handler(500, internal_server_error)

    app.run(debug=True, port=5000)