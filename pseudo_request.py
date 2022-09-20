import requests
import torch
import numpy as np

# url = 'http://localhost:5000/api'
url = 'http://localhost:5001/api'

x = torch.randn(256, 32)
# print(x)
payload = x.tolist()
# print(len(payload))
# print(x.shape)
response = requests.post(url, json={
							"eeg": payload,
							# "timestamp": timestamp
						})
print(response.text)