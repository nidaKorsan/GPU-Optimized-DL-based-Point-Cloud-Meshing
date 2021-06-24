import numpy as np
from tensorflow.keras.utils import Sequence
import cv2


class DataGenerator(Sequence):
    def __init__(self, path, dataSize, batch_size=32, dim=(360,480,3),shuffle=True,buffer_size =32,augment =True, offset=0):
        
        self.dim = dim
        self.batch_size = batch_size
        self.path = path
        self.dataSize = dataSize
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.buffer_index = 0
        self.augment = augment
        self.offset = offset
        if buffer_size > 0:
            self.buffer =  np.empty((buffer_size*self.batch_size, *self.dim), dtype=np.uint8)
            self.label_buffer = np.empty((buffer_size*self.batch_size, *self.dim), dtype=np.uint8)


    def __len__(self):
        return int(np.floor(self.dataSize / self.batch_size))

    def reload_buffer(self):
        buffer_length = self.buffer_size*self.batch_size
        if buffer_length * (self.buffer_index + 1) > self.dataSize:
            buffer_length = self.dataSize % buffer_length
        for i in range(buffer_length):
            target = self.buffer_index*buffer_length+i+self.offset
            self.buffer[i,] = cv2.imread(self.path + "inputs\\" + str(target) + '.png')
            self.label_buffer[i,] = cv2.imread(self.path + "outputs\\" + str(target) + '.png')
            
        self.buffer_index +=1
        

    def __getitem__(self, index):

        if index == 0 or index % self.buffer_size == 0:
            self.reload_buffer()

        target_index = 0 if index == 0 else index % (self.buffer_size)
    
        end = (target_index+1)*self.batch_size
        
        if end >= len(self.buffer):
            end = len(self.buffer); 
        
        X = self.buffer[target_index*self.batch_size:end]

        y = self.label_buffer[target_index*self.batch_size:end]
        #y_ret = [i[0] for i in y]
        #y_ret = np.array(y_ret).reshape((self.batch_size, 3*480))
        return X / 255, np.array(y[:][0:100])/255


    def on_epoch_end(self):
        self.reset()

    def reset(self):
        self.buffer_index = 0
