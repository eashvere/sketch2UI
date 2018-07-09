import json
import torch
from PIL import Image
import flask
from flask import render_template
from simple_request import predict_result

app = flask.Flask(__name__)

@app.route("/")
def home_screen():
    render_template('home.html')

@app.route('/predict')
def predict():
    




if __name__ == '__main__':
    app.run(debug=True)