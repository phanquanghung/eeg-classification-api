# EEG Classification API

## Introduction
**EEG Classification API** is an API for classifying EEG data, especially for projects belonging to HMI Laboratory, VNU-UET.

### Motor Imagery Classification model ([EEGNetv4](https://arxiv.org/abs/1611.08024))
#### Parameters
- `n_classes` : number of classes.
- `in_chans` : number of channels fed to model.
- `input_window_samples` : total number of data points from a channel in one 'trial' - an input to get an output from the API.

Therefore, the dimension of my input is `in_chans` x `input_window_samples` - 32x256.

#### Pre-trained Weight
It is not necessary to train the model because pre-trained weight, `three_classes.pth`, is already included in the repo. 

### API
> **Warning**
> 
> This code is converted from notebook, so apprehently it is not ideal for reproducibility, or production. This must be carefully examined before being put into use.

The API is divided into two parts, `server.py` and `request.py`. The server loads the model and handles POST requests, and then returns the results to the client side.

While it should be noted that the input data must include full (32) channels, the transfer option by chunk (32) is also recommended. If implemented correctly, the model will make a prediction approximately every 2 seconds.

## Getting Started

### Requirements
To reproduce exact results, I recommend installing an environment identical to mine.

First, create a new conda environment with
```
conda create --name myenv python=3.10.4
conda activate myenv
```
Then, install the remaining dependencies:
```
pip install -r requirements.txt
```

### Usage
#### Running the server

Launch the server on a separate terminal from request.py with:

```
python server.py
```

#### Sending requests

While the `server.py` file is complete and requires almost no editing to run, `request.py` file was written as minimally as possible so that installers can use it as a template when coding their own requests.

```
python request.py
```

If not using Lab Streaming Layer, installer can experiment with random numbers in `pseudo_request.py` file.

```
python pseudo_request.py
```
