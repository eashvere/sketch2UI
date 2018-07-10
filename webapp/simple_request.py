# encoding: utf-8
# Borrowed from https://github.com/L1aoXingyu/deploy-pytorch-model

import requests
import argparse

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'


def predict_result(image):
    # Initialize image path
    payload = {'image': image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        labels = []
        boxes = []
        flipped = []
        for image in r['prediction']:
            labels.append(image['label'])
            boxes.append(image['bounding_boxes'])
            flipped.append(image['flipped'])
        return labels, boxes, flipped
    # Otherwise, the request failed.
    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')

    args = parser.parse_args()
    predict_result(args.file)
