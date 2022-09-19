import requests
from pylsl import StreamInlet, resolve_stream

url = 'http://localhost:5000/api'

def send_eeg():
	streams = resolve_stream('name', 'EmotivDataStream-EEG')

	# # create a new inlet to read from the stream
	inlet = StreamInlet(streams[0])

	while True:
		sample, timestamp = inlet.pull_chunk()
		if len(timestamp) > 0:
			response = requests.post(url, json={
										"eeg_data": sample,
										"timestamp": timestamp
									})
			print(response.text)

if __name__ == '__main__':
	send_eeg()