# EEG Classification API

> **Warning**
> 
> This code is converted from different notebooks, so apparently it is not ideal for reproducibility, or production. This must be carefully examined before being put into use.

## Introduction
**EEG Classification API** is an API for classifying EEG motor imagery data, using [EEGNet](https://arxiv.org/abs/1611.08024) architecture. It was created especially for projects belonging to HMI Laboratory, VNU-UET.

### Classification Models

The API consists of two models implemented by us, model #1 (by @phanquanghung) and model #2 (by @txdat).

#### Parameters
- `n_classes` : number of classes.
- `in_chans` : number of channels fed to model.
- `input_window_samples` : total number of data points from a channel in one 'trial' - an input to get an output from the API.

The dimension of both models' input is `in_chans` x `input_window_samples`.

1. **Model #1**
- Classes (`n_classes` = 3): left hand, right hand & other
- Dimension `in_chans` x `input_window_samples`:  **32** x 256.

2. **Model #2**
- Classes (`n_classes` = 4): leg, right hand, left hand & rest
- Dimension `in_chans` x `input_window_samples`: **28** x 256. 
Four electrodes are omitted in the setup of this model, including `['FT9', 'PO9', 'PO10', 'FT10']` .

> **Warning**
> 
> Model #2 was passed into the API without being tested for correctness. Users should exercise with extreme caution and compare with the [original model repository](https://github.com/txdat/bci-motor-imagery/blob/master/notebooks/eeg_final.ipynb) if necessary. In other words, consider this model #2's code just to demonstrate its usability.

#### Pre-trained Weight
It is not necessary to train the models because pre-trained weights for both models, `three_classes.pth` (for model #1) and `EEGNet8,4_nonEA.ckpt` (for model #2), are already included in the repo. 

### API

The API is divided into two parts, **server** and **request**. The server loads the model, handles POST requests, and then returns the results to the client.

While it should be noted that the input data must include total (32) channels, the transfer option by chunk (32) is also recommended. If implemented correctly, the server will return a prediction approximately every 2 seconds.

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

> **Note**
> 
> The default ports for server with models #1 and #2 are `5000` and `5001` respectively. Make sure you have set the correct port in the `request.py` file.

Launch the server with model #1 on a separate terminal from request.py with:

```
python server_3classes.py
```

Or the server with model #2 with:

```
python server_4classes.py
```

#### Sending requests

While the server is somewhat complete and requires almost no editing to run, the `request.py` file was written as minimally as possible so that users can use it as a template when coding their own requests. 

Run `request.py` if you are using an EMOTIV device (which transmits data through the Lab Streaming Layer) with:

```
python request.py
```

If not using Lab Streaming Layer, the user can experiment with random numbers from the `pseudo_request.py` file with:

```
python pseudo_request.py
```

---

**Authors:**
- @phanquanghung - API & classification model #1
- @txdat - classification model #2
