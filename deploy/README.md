# A Simple PyTorch deep learning REST API

This repository contains the code of [如何用flask部署pytorch模型](https://zhuanlan.zhihu.com/p/35879835)

## Requirements

Before you run these files, please install your Redis Server


### Linux
On Linux, use these [installation instructions](https://redis.io/download)


### MacOS
On Mac, install [Homebrew](https://brew.sh/) and then run 

```bash
brew install redis
brew services start redis
redis-cli ping
```
 and check if PONG is returned

 ### Windows

 Unfortunately, Redis does not have good support on Windows so instead use checkout the single-threaded branch and run that instead

## Starting the pytorch server

```bash
python run_pytorch_server.py 
```

You can now access the REST API via `http://127.0.0.1:4000/predict`

## Submitting requests to pytorch server

Go to the webapp folder and run

```bash
python app.py
```

The website to connect to will default to `http://127.0.0.1:5000`

## Acknowledgement
This repository refers to [jrosebr1/simple-keras-rest-api](https://github.com/jrosebr1/simple-keras-rest-api), and thank the author again.