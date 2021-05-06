from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse, JsonResponse
from camera.camera_helper import WebCam
import time
import tensorflow as tf
import numpy as np
import cv2
import os
import statistics
from statistics import mode

#Global variables
freq = []
predictedString = ''
predictedCharacter = ''
cameraStarted = False

#Get TF lite model
SIZE = 64, 64
PATH = os.getcwd() + "\camera\Alphabet_Classifier__Preprocess_Lite.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#index to character mapping
map_idx_to_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
                   6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                   12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                   18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                   24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: ' ',
                   29: 'other'}

#Return most common character from list
def most_common(List):
    return(mode(List))

#Index page, endpoint for Ajax request
def index(request):
	global cameraStarted
	global predictedString

	if request.method == 'GET':
		cameraStarted = False
	else:
		cameraStarted = not cameraStarted
		predictedString = ''

	return render(request, 'index.html', {'cameraStarted': cameraStarted})

def getPredictions(image):
	return JsonResponse({
		'predictedCharacter': predictedCharacter,
		'predictedString': predictedString
	})

#To generate camera image and make predictions
def gen(camera):
	global freq
	global predictedString
	global predictedCharacter

	while True:
		frame, image = camera.get_frame()
		predictedCharacter = prediction(image)
		if predictedCharacter!='nothing':
			nothingCtr = 0
			freq.append(predictedCharacter)
		else:
			if freq != []:
				predictedString += most_common(freq)
				freq = []
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
	return redirect('index.html')
	

def video_feed(request):
	return StreamingHttpResponse(gen(WebCam()),
					content_type='multipart/x-mixed-replace; boundary=frame')


def prediction(image):
	
	#Preprocessing
	resizedImage = cv2.resize(image, SIZE)
	input_data = np.array(resizedImage, dtype = np.float32)
	input_data = input_data[np.newaxis, ...]
	interpreter.set_tensor(input_details[0]['index'], input_data)

	interpreter.invoke()

	# The function `get_tensor()` returns a copy of the tensor data.
	# Use `tensor()` in order to get a pointer to the tensor.
	output_data = interpreter.get_tensor(output_details[0]['index'])
	idx = np.argmax(output_data, axis=-1)
	prediction = map_idx_to_char.get(idx[0])

	return prediction
