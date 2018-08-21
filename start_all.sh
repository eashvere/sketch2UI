#!/bin/sh

python deploy/classify_process.py &
python deploy/run_pytorch_server.py &
python webapp/app.py &