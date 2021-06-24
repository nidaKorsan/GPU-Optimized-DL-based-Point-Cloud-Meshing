import time
import zmq
import numpy as np
from tensorflow import keras

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Loading model...")
reconstructed_model = keras.models.load_model("C:/Dataset/saves_final/", compile=False)
print("Model loaded.")

while True:

    #  Wait for next request from client
    print("Waiting for input...")
    message = socket.recv() #recieve input array
    print("Input array has been received.")
    input_array = np.frombuffer(message, dtype=np.float64)
    input_array = input_array.reshape((1,20,20,3))
    print("INPUT ARRAY: ", len(input_array), input_array.shape)
    #print(*input_array, sep='\n')
    output = reconstructed_model.predict(input_array)
    print("Output has been predicted.")
    print("Shape of output ", output.shape, type(output[0][0]))
    #  Send reply back to client
    socket.send(output[0].tobytes())#send output