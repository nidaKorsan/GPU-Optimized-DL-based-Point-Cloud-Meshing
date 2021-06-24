from DataGenerator import DataGenerator
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import cv2
import sys
import math

@tf.function
def surrogate(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return 0.1* tf.abs(y_true - y_pred) + tf.square(y_true - y_pred)

def scheduler(epoch, lr):
     return lr * math.exp(-0.03)

def  getCheckPointCallback(folder): 
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=folder + "/saves_final",
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    return model_checkpoint_callback


def getModel(input_shape):
   ### Create Model
   input_s = Input((20,20,3))
   backbone = tf.keras.applications.MobileNetV2(input_tensor = input_s, alpha=1.0, include_top=False, weights=None)
   lastLayer = backbone.get_layer("block_13_project")
   flatten = tf.keras.layers.Flatten()(lastLayer.output)
   newLayer = Dense(654,activation ="relu",name ="intmdt1")(flatten)
   newLayer1 = Dense(768,activation="relu",name ="intmdt2")(newLayer)
   newLayer2 = Dense(768,activation="relu",name ="intmdt3")(newLayer1)
   finalLayer = Dense(768,activation="relu",name ="intmdt4")(newLayer2)
   projection_head = Dense(400*200,activation ="linear",name ="final_layer",)(newLayer)
   #outputs = layers.Conv2D(3,kernel_size= 3, activation="softmax", name ="final_layer", padding="same")(lastLayer)
   model = tf.keras.Model(inputs=backbone.inputs,outputs=projection_head)
   
   return model


def createDataGenerators(path, batch_size,buffer_size, trainCount, testCount):
    print("Creating Data Generators")
 
    datagen = DataGenerator(path, dataSize= trainCount, batch_size=batch_size, buffer_size =  int(trainCount / (trainCount + testCount) * buffer_size))
    valdatagen = DataGenerator(path, dataSize = testCount,batch_size=batch_size, buffer_size = int(testCount / (trainCount + testCount) * buffer_size))
    return datagen, valdatagen

# def trainModel(model,datagen, valdatagen, epochs, callback_functions):
    # print("Starting training for " + str(epochs))
    # myhistory =[]
    # FINE_TUNE_LAYER = 25
    # for m in range(FINE_TUNE_LAYER-5,FINE_TUNE_LAYER):
        # model.layers[m].trainable =True
    # history = model.fit(datagen,validation_data =valdatagen,callbacks=[callback_functions], epochs=5,shuffle = True, batch_size = 1)
    # 
    # myhistory.append(history)
    # for m in range(FINE_TUNE_LAYER-10,FINE_TUNE_LAYER-5):
        # model.layers[m].trainable =True
    # history = model.fit(datagen,validation_data =valdatagen,callbacks=[callback_functions], epochs=5,shuffle = True, batch_size = 8)
    # myhistory.append(history)
    # 
    # for m in range(FINE_TUNE_LAYER-15,FINE_TUNE_LAYER-10):
        # model.layers[m].trainable =True
    # history = model.fit(datagen,validation_data =valdatagen,callbacks=[callback_functions,tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)], epochs=10,shuffle = True, batch_size = 8)
    # myhistory.append(history)
    # 
    # for m in range(FINE_TUNE_LAYER-20,FINE_TUNE_LAYER-15):
        # model.layers[m].trainable =True
    # history = model.fit(datagen,validation_data =valdatagen,callbacks=[callback_functions,tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)], epochs=15,shuffle = True, batch_size = 8)
    # myhistory.append(history)
    #  
    # for m in range(0,FINE_TUNE_LAYER-20):
        # model.layers[m].trainable =True
    # history = model.fit(datagen,validation_data =valdatagen,callbacks=[callback_functions,tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)], epochs=epochs,shuffle = True, batch_size = 8)
    # myhistory.append(history)
    # return myhistory

def trainModel(model,d1, d2, l1, l2, epochs, callback_functions):
    print("Starting training for " + str(epochs))
    myhistory =[]
    FINE_TUNE_LAYER = 25
    for m in range(FINE_TUNE_LAYER-5,FINE_TUNE_LAYER):
        model.layers[m].trainable =True
    print("Shapes", d1.shape, d2.shape, l1.shape, l2.shape)
    
    history = model.fit(d1,l1,validation_data =(d2,l2),callbacks=[callback_functions], epochs=5,shuffle = True)
    
    myhistory.append(history)
    for m in range(FINE_TUNE_LAYER-10,FINE_TUNE_LAYER-5):
        model.layers[m].trainable =True
    history = model.fit(d1,l1,validation_data =(d2,l2),callbacks=[callback_functions], epochs=5,shuffle = True, batch_size = 8)
    myhistory.append(history)
    
    for m in range(FINE_TUNE_LAYER-15,FINE_TUNE_LAYER-10):
        model.layers[m].trainable =True
    history = model.fit(d1,l1,validation_data =(d2,l2),callbacks=[callback_functions,tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)], epochs=10,shuffle = True, batch_size = 8)
    myhistory.append(history)
    
    for m in range(FINE_TUNE_LAYER-20,FINE_TUNE_LAYER-15):
        model.layers[m].trainable =True
    history = model.fit(d1,l1,validation_data =(d2,l2),callbacks=[callback_functions,tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)], epochs=15,shuffle = True, batch_size = 8)
    myhistory.append(history)
     
    for m in range(0,FINE_TUNE_LAYER-20):
        model.layers[m].trainable =True
    history = model.fit(d1,l1,validation_data =(d2,l2),callbacks=[callback_functions,tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)], epochs=epochs,shuffle = True, batch_size = 8)
    myhistory.append(history)
    return myhistory
def dataRead(path, trainCount, testCount):
    d1 = np.empty((trainCount, 360,480,3), dtype=np.uint8)
    d2 = np.empty((testCount, 360,480,3), dtype=np.uint8)
    l1 = np.empty((trainCount, 480*360*3), dtype=np.uint8)
    l2 = np.empty((testCount, 480*360*3), dtype=np.uint8)
    print("Directly loading to memory")
    print("first ", d1[0].nbytes)
    for m in range(trainCount):
        d1[m] = cv2.imread(path+ "inputs\\" + str(m)+".png")
    for m in range(trainCount, trainCount + testCount):
        d2[m - trainCount] = cv2.imread(path+ "inputs\\" + str(m)+".png")
    for m in range(trainCount):
        l1[m] = cv2.imread(path+ "outputs\\" + str(m)+".png").reshape(480*360*3)
    for m in range(trainCount, trainCount + testCount):
        l2[m-trainCount] = cv2.imread(path+ "outputs\\" + str(m)+".png").reshape(480*360*3)
    print(len(d1))
    print("second ", d1[0].nbytes)
    print(0 in d1[0])

    return d1, l1, d2, l2

def readData(path, trainCount, testCount):
    d1 = np.empty((trainCount, 20,20,3))
    d2 = np.empty((testCount, 20,20,3))
    l1 = np.empty((trainCount, 400*200), dtype=np.uint8)
    l2 = np.empty((testCount, 400*200), dtype=np.uint8)

    for m in range(trainCount):
        a = np.loadtxt(path + "inputs/" + str(m)+".nk").reshape((20,20,3))
        d1[m] = a#np.pad(a, ((0,12), (0,12), (0,0)), 'constant', constant_values=(0))
    for m in range(trainCount, trainCount + testCount):
        a = np.loadtxt(path + "inputs/" + str(m)+".nk").reshape((20,20,3))
        d2[m - trainCount] = a#np.pad(a, ((0,12), (0,12), (0,0)), 'constant', constant_values=(0))

    for m in range(trainCount):
        l1[m] = np.loadtxt(path + "outputs/" + str(m)+".vector")
        l1 = l1 == 1
    print("Did the training set")
    for m in range(trainCount, trainCount + testCount):
        l2[m - trainCount] = np.loadtxt(path + "outputs/" + str(m)+".vector")
        l2 = l2 == 1

    return d1, d2, l1,  l2
#    hue = 0
#    for i in l1:
#        k = 0
#        m = 1
#        print("hue : " , hue)
#        hue += 1
#        for j in range(len(i)):
#            edge = i[j]
#            if edge == 1:
#                if not((m == k - 1 or m == k+1) or \
#                (m - 20 == k or m + 20 == k) or \
#                ( m - 21 == k or m - 19 == k) or \
#                ( m + 21 == k or m + 19 == k)):
#                    print("thats ma man")
#
#            m += 1
#            if m == 400:
#                k += 1
#                m = k + 1
                

print("Reading Data")
#readData("C:/Dataset/", 500, 100)
d1, d2, l1, l2 = readData("C:/Dataset/", 50, 10)

print("Creating Model")
input_shape = (32, 32, 3)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model  = getModel(input_shape)
model.summary()
model.compile(optimizer =optimizer,loss = keras.losses.BinaryCrossentropy(from_logits=True))
#datagen, valdatagen = createDataGenerators("C:\\Renders\\", 8, 1024, 80, 24)
callBack = getCheckPointCallback("C:/Dataset/")
#print("Shapes", d1.shape, d2.shape, l1.shape, l2.shape)
#model.save('C:/Dataset/my_model')
history = trainModel(model, d1, d2, l1, l2, 20, callBack)
print("Training has finished")

# print("Saving history")

# for m in range(len(history)):
#     np.save("C:\\Renders\\training_history"+str(m) +  ".npy",history[m].history)

#print(cv2.imread("C:\\Renders\\"+ "inputs\\" + str(0)+".png"))