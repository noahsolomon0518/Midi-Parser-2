from midi_parser.network_optimizer import GridSearch
from midi_parser.decimal_encoders import DecimalEncoderOnOff
from midi_parser.data_generators import DataGenOnOff
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

encoder = DecimalEncoderOnOff("test/test_data/midis", 1/64, 50)
datagenTrain = DataGenOnOff(encoder)
datagenTest = DataGenOnOff(encoder)
print(datagenTest[0])



from unittest import TestCase
import unittest

nClasses = 148
lookback =  50


def buildModelFn(layers, nodes, dropout):
    model = Sequential()
    
    model.add(LSTM(nodes, return_sequences = (layers==2), input_shape = (lookback, nClasses)))
    if(layers ==2 ):
        model.add(LSTM(nodes))
    if(dropout>0):
        model.add(Dropout(dropout))
    model.add(Dense(148, activation="softmax"))
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=["accuracy"])
    return model
    



class TestNetworkOptimizer(TestCase):


    def test_grid_search(self):
        grid = GridSearch(buildModelFn, dict(layers = [1,2], nodes = [64,128,256], dropout = [0,0.25,0.5]))
        grid.fit(datagenTrain, datagenTest, path = "test/test_data/testlog.csv", epochs = 1)