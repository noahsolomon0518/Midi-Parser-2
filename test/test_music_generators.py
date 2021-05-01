from unittest import TestCase
import unittest
from midi_parser.data_generators import DataGenOnOff, DataGenMultiNet, DataGenEmbeddedMultiNet
from midi_parser.music_generators import GeneratorOnOff, GeneratorMultiNet, GeneratorEmbeddedMultiNet
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet, DecimalEncoderMultiNet2
from midi_parser.music_generators import sampleTop

from keras.models import load_model


lookback = 100
nClassesTimes = 40
smallestTimeUnit = 1/64
noteRange = (36,84)
nClassesNotes = 2*(noteRange[1] - noteRange[0] + 1)


'''
### For OnOff
from keras.models import Sequential
from keras.layers import Dense, LSTM
encoder = DecimalEncoderOnOff("test/test_data/midis", 1/64, nClassesTimes, noteRange= noteRange)
datagen = DataGenOnOff(encoder, lookback = lookback)
model = Sequential()
model.add(LSTM(64, input_shape = (lookback, nClassesNotes + nClassesTimes)))
model.add(Dense(nClassesTimes+nClassesNotes ,activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(datagen, epochs = 40)
model.save("test/test_data/models/test_model_on_off.h5")


### For MultiNet

from keras.models import Model, Input
from keras.layers import Dense, LSTM, concatenate
encoderMultiNet = DecimalEncoderMultiNet("test/test_data/midis", 1/64, nClassesTimes, noteRange= noteRange)
datagenMultiNet = DataGenMultiNet(encoderMultiNet, lookback = lookback)
nClassesNotesMultiNet = 48+1+1
nClassesTimesMultiNet = 40
nClassesMultiNet = nClassesTimesMultiNet + nClassesNotesMultiNet 

modelIn = Input(shape = (lookback, nClassesMultiNet))

noteNet = LSTM(64)(modelIn)
noteNet = Dense(nClassesNotesMultiNet, activation="softmax")(noteNet)

timeNet = LSTM(64)(modelIn)
timeNet = concatenate([timeNet, noteNet])
timeNet = Dense(nClassesTimesMultiNet, activation="softmax")(timeNet)

model = Model(modelIn,[noteNet, timeNet])

model.compile(optimizer = 'rmsprop', loss = ['categorical_crossentropy','categorical_crossentropy'], metrics = ['accuracy'])
model.fit(datagenMultiNet, epochs = 40)


model.save("test/test_data/models/test_model_multi_net.h5")



### For MultiNet2

from keras.models import Model, Input
from keras.layers import Dense, LSTM, concatenate
encoderMultiNet2 = DecimalEncoderMultiNet2("test/test_data/midis", 1/64, nClassesTimes, noteRange= noteRange)
datagenMultiNet2 = DataGenMultiNet2(encoderMultiNet2, lookback = lookback)
nClassesNotesMultiNet = 48+1+1
nClassesTimesMultiNet = 40
nClassesMultiNet = nClassesTimesMultiNet + nClassesNotesMultiNet 

modelIn = Input(shape = (lookback, nClassesMultiNet))

noteNet = LSTM(64)(modelIn)
noteNet = Dense(nClassesNotesMultiNet, activation="softmax")(noteNet)

timeNet = LSTM(64)(modelIn)
timeNet = concatenate([timeNet, noteNet])
timeNet = Dense(nClassesTimesMultiNet, activation="softmax")(timeNet)

model = Model(modelIn,[noteNet, timeNet])

model.compile(optimizer = 'rmsprop', loss = ['categorical_crossentropy','categorical_crossentropy'], metrics = ['accuracy'])
model.fit(datagenMultiNet2, epochs = 10)


model.save("test/test_data/models/test_model_multi_net2.h5")


'''


class TestMusicGeneratorOnOff(TestCase):
    
    encoder = DecimalEncoderOnOff("test/test_data/midis", 1/64, nClassesTimes, noteRange= noteRange)
    datagen = DataGenOnOff(encoder, lookback = lookback)
    model = load_model("test/test_data/models/test_model_on_off.h5")


    def test_init(self):
        musicGen = GeneratorOnOff(TestMusicGeneratorOnOff.model, TestMusicGeneratorOnOff.datagen)
    
    
    def test_generate(self):
        musicGen = GeneratorOnOff(TestMusicGeneratorOnOff.model, TestMusicGeneratorOnOff.datagen)
        musicGen.generate(1,100).play()



encoderMultiNet = DecimalEncoderMultiNet("test/test_data/midis", 1/64, nClassesTimes, noteRange= noteRange)
datagenMultiNet = DataGenMultiNet(encoderMultiNet, lookback = lookback)
nClassesNotesMultiNet = 48+1+1
nClassesTimesMultiNet = 40
nClassesMultiNet = nClassesTimesMultiNet + nClassesNotesMultiNet 



class TestMusicGeneratorMultiNet(TestCase):
    encoder = DecimalEncoderMultiNet("test/test_data/midis", 1/64, nClassesTimesMultiNet, noteRange= noteRange)
    datagen = DataGenMultiNet(encoderMultiNet, lookback = lookback)
    model = load_model("test/test_data/models/test_model_multi_net.h5")


    def test_init(self):
        musicGen = GeneratorMultiNet(TestMusicGeneratorMultiNet.model, TestMusicGeneratorMultiNet.datagen)
    
    
    def test_generate(self):
        musicGen = GeneratorMultiNet(TestMusicGeneratorMultiNet.model, TestMusicGeneratorMultiNet.datagen)
        musicGen.generate(1,100).play()

    def test_generate_top_probs(self):
        musicGen = GeneratorMultiNet(TestMusicGeneratorMultiNet.model, TestMusicGeneratorMultiNet.datagen)
        musicGen.generate(1,100, 3).play()



'''
from keras.models import Model, Input
from keras.layers import Dense, LSTM, concatenate, Embedding

noteIn = Input(shape = (lookback,))
noteEmbed = Embedding(nClassesNotesMultiNet, 20, input_length=lookback)(noteIn)

timeIn = Input(shape = (lookback,))
timeEmbed = Embedding(nClassesTimesMultiNet, 20, input_length=lookback)(timeIn)

concated = concatenate([noteEmbed, timeEmbed])

lstmNote = LSTM(64)(concated)
noteOut = Dense(nClassesNotesMultiNet, activation = 'softmax')(lstmNote)



lstmTime = LSTM(64)(concated)
timeOut = concatenate([noteOut, lstmTime])
timeOut = Dense(nClassesTimesMultiNet, activation = "softmax")(timeOut)

modelEmbed = Model([noteIn, timeIn],[noteOut,timeOut])
modelEmbed.compile(optimizer = 'rmsprop', loss = ['categorical_crossentropy','categorical_crossentropy'], metrics = ['accuracy'])

modelEmbed.fit(datagenEmbeddedMultiNet , epochs = 10)
modelEmbed.save("test/test_data/models/test_model_embedded_multinet.h5")
'''
encoderMultiNet2 = DecimalEncoderMultiNet2("test/test_data/midis", 1/64, nClassesTimes, noteRange= noteRange)
datagenEmbeddedMultiNet = DataGenEmbeddedMultiNet(encoderMultiNet2, lookback = lookback)
nClassesNotesMultiNet = 48+1+1
nClassesTimesMultiNet = 40
modelEmbed = load_model("test/test_data/models/test_model_embedded_multinet.h5")
class TestMusicGeneratorEmbeddedMultiNet(TestCase):

    def test_generate(self):
        musicGen = GeneratorEmbeddedMultiNet(modelEmbed, datagenEmbeddedMultiNet, loo)
        musicGen.generate(temp = 1, nNotes = 100, sampleTopProbs = 3).play()




class Sampler(TestCase):
    
    
    
    def test_sampling(self):
        for i in range(30):
            top = sampleTop([[0.4,0.1,0.01,0.09,0.4]], 3)
            self.assertGreaterEqual(top, 0.1)
        










    
        




#data cleaning

import numpy as np 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class DataEncoder:
    def __init__(self, df, normalize = True):
        self.df = df.copy()
        self.normalize = normalize
        self.ohe = None
        self.scaler = None
        
        
    def oneHotEncode(self, dfOneHot):
        self.ohe = OneHotEncoder(sparse = False)
        return self.ohe.fit_transform(dfOneHot)
    
    def _normalize(self, df):
        self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(df)
        
    
    def encode(self, ordinalCols, oneHotCols, yCol):
        #Used in inverseTransform
        self.ordinalCols = ordinalCols
        self.oneHotCols = oneHotCols
        self.yCol = yCol
        
        
        dfOrdinal = self.df.loc[:,ordinalCols].to_numpy()
        dfOneHot = self.oneHotEncode(self.df.loc[:,oneHotCols])
        dfY = self.df.loc[:,yCol].to_numpy().reshape(-1,1)
        self.dfEncoded = np.concatenate([dfOrdinal,dfOneHot, dfY], axis = 1)
        
        if(self.normalize):
            self.dfEncoded = self._normalize(self.dfEncoded)
            
        return self.dfEncoded
        
    #Can pass in cleaned data points and convert back to categories    
    def inverseTransform(self, samples, isY):
        if(self.ohe == None):
            raise Exception("DataEncoder has not encoded data yet.")
        samples = np.array(samples)
        if(len(samples.shape)==1):
            samples = samples.reshape(1,-1)
              
      
        arrOrdinal = samples[:,:len(self.ordinalCols)]
        if(isY):
            arrOneHot = samples[:,len(self.ordinalCols):-1]
            arrOneHotUnEncoded = self.ohe.inverse_transform(arrOneHot)
            arrY = samples[:,-1].reshape(-1,1)
            arrUnEncoded = np.concatenate([arrOrdinal, arrOneHotUnEncoded, arrY], axis = 1)
            if(self.normalize):
                arrUnEncoded = self.scaler.inverse_transform(arrUnEncoded)
            return arrUnEncoded = pd.DataFrame(arrUnEncoded, columns = np.concatenate([self.ordinalCols, self.oneHotCols, [self.yCol]]))
        
        else:
            arrOneHot = samples[:,len(self.ordinalCols):]
            arrUnEncoded = np.concatenate([arrOrdinal, self.ohe.inverse_transform(arrOneHot)],axis = 1)


            if(self.normalize):
                arrUnEncoded = self.scaler.inverse_transform(arrUnEncoded)
            return arrUnEncoded = pd.DataFrame(arrUnEncoded, columns = np.concatenate([self.ordinalCols, self.oneHotCols]))