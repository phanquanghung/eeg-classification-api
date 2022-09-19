# Import libraries
import torch
import torch.nn.functional as F
from braindecode.models import EEGNetv4

import numpy as np
from flask import Flask, request

app = Flask(__name__)

cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# rng = RandomState(seed)

# Load the model
model = EEGNetv4(
    in_chans = 32,
    n_classes = 3,
    input_window_samples=256,
)

checkpoint = torch.load('three_classes.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# if cuda:
#     input = input.cuda()
#     model = model.cuda()
model.eval()

@app.route('/api',methods=['POST'])

def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    # prediction = model.predict([[np.array(data['exp'])]])
    # return str(type(data['eeg']))
    input_server = np.array(data['eeg'])
    input_tensor = torch.from_numpy(input_server).float()

    output = model(input_tensor)
    output = F.softmax(output, dim=1)
    
    if output[0,0] == torch.max(output):
        return 'left'
    elif output[0,1] == torch.max(output):
        return 'right'
    elif output[0,2] == torch.max(output):
        return 'other' 
    # return 'collecting data'

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")