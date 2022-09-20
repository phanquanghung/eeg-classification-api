# Import libraries
import torch
import torch.nn.functional as F
from braindecode.models import EEGNetv4

import numpy as np
from flask import Flask, request

app = Flask(__name__)

# cuda = torch.cuda.is_available()
# print('gpu: ', cuda)
# device = 'cuda' if cuda else 'cpu'

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# rng = RandomState(seed)

buffer = np.empty((0, 0))

# def load_model():
# Load the model
model = EEGNetv4(
    in_chans = 32,
    n_classes = 3,
    input_window_samples=256,
)
checkpoint = torch.load('three_classes.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
# if cuda:
#     input = input.cuda()
#     model = model.cuda()
model.eval()

def update_buffer(data_np):
    global buffer
    if buffer.shape[0] == 0:
        buffer = data_np
    else:
        buffer = np.append(buffer, data_np, axis = 0)

def preprocess(full_buffer):
    # add batch dims
    full_buffer = np.expand_dims(full_buffer, axis=0)
    full_buffer = np.expand_dims(full_buffer, axis=0)
    data_tensor = torch.from_numpy(full_buffer).float()
	# convert type to torch tensor
	# # input = torch.from_numpy(input)
	# reshape input size
    data_tensor = data_tensor.permute(1, 3, 2, 0)
    return data_tensor

def predict(input):
    output = model(input)
    output = F.softmax(output, dim=1)
    return output  

@app.route('/api',methods=['POST'])
def update_status():
    global buffer
    # Get the data from the POST request.
    data = request.get_json(force=True)
    data_np = np.array(data['eeg'])
    data_np = data_np.astype('float32') * 1e3

    update_buffer(data_np)
    if buffer.shape[0] == 256:
        input = preprocess(buffer)
        buffer = np.empty((0, 0))
        output = predict(input)
        if output[0,0] == torch.max(output):
            return 'left'
        elif output[0,1] == torch.max(output):
            return 'right'
        elif output[0,2] == torch.max(output):
            return 'other'
    else:
        return 'collecting data'
    # input_cat = torch.cat((input_cat, input_tensor), 0)


if __name__ == '__main__':
    # load_model()
    # try:
        app.run(port=5000, debug=True)
    # except:
    #     print("Server is exited unexpectedly. Please contact server admin.")