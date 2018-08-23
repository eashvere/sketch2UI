#!/bin/sh

trap 'kill %1; kill %2' SIGINT
python deploy/classify_process.py & python deploy/run_pytorch_server.py & python webapp/app.py