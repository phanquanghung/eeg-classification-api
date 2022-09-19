import requests
import torch
import numpy as np

import time

url = 'http://localhost:5000/api'

x_val = np.load('/home/ducanh/hain/hungp/NeurIPS_BEETL/numpy_data_binary/X_val.npy')

trial_id = 0

for x in range(10):
	# original shape: (num_trial, num_electrons, num_samples) with each trial is 1 label
	input = x_val[x, :, :]
	input = input.astype('float32') * 1e3
	# print(input.shape)

	# add batch dims
	input = np.expand_dims(input, axis=0)
	input = np.expand_dims(input, axis=0)

	# convert type to torch tensor
	input = torch.from_numpy(input)
	# reshape input size
	input = input.permute(1, 2, 3, 0)	

	# print(input.shape)
	# payload = {
	# 	'exp':1.8
	# }

	payload = input.tolist()
	response = requests.post(url, json={
								"eeg": payload,
								# "timestamp": timestamp
							})
	print(response.text)
	# r = requests.post(url,json=payload)
	# print(r.text)
	time.sleep(1)