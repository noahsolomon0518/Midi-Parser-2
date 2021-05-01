from unittest import TestCase
import unittest
from midi_parser.pieces import OnOffPiece, MultiNetPiece
from midi_parser.data_generators import DataGenOnOff, encodeFromOneHot, DataGenMultiNet, DataGenEmbeddedMultiNet
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet, DecimalEncoderMultiNet2

encoder = DecimalEncoderOnOff("test/test_data/midis", 1/64, 40)
encoderMultiNet = DecimalEncoderMultiNet("test/test_data/midis", 1/64, 40)
encoderMultiNet2 = DecimalEncoderMultiNet2("test/test_data/midis", 1/64, 40)

print("YERT")
class TestDataGeneratorOnOff(TestCase):
    

    def test_init(self):
        datagen = DataGenOnOff(encoder,50)
        

    def test_get_data(self):
        datagen = DataGenOnOff(encoder, 10,  lookback=50)
        data = datagen.__getitem__(0)
    

    def test_play_sample(self):
        datagen = DataGenOnOff(encoder, 10,  lookback=100)
        data = datagen.__getitem__(0)
        piece = OnOffPiece(datagen.ohe.inverse_transform(data[0][0]).reshape(-1).tolist(), 1/64)
        piece.play()



class TestDataGeneratorMultiNet(TestCase):
    

    def test_init(self):
        datagen = DataGenMultiNet(encoderMultiNet,50)
        

    def test_get_data(self):
        datagen = DataGenMultiNet(encoderMultiNet, 10,  lookback=50)
        x, [yNotes, yTimes] = datagen.__getitem__(0)
        print(x.shape)
        print(yNotes.shape)
        print(yTimes.shape)

    def test_play_samples(self):
        datagen = DataGenMultiNet(encoderMultiNet, 10,  lookback=50)
        for i in range(5):
            x, [yNotes, yTimes] = datagen.__getitem__(i)
            pieceNotes = datagen.ohe.inverse_transform(x[0])[:, 0]
            pieceTimes = datagen.ohe.inverse_transform(x[0])[:, 1]
            piece = MultiNetPiece([pieceNotes,pieceTimes], 1/64)
            piece.play()





class TestDataGeneratorMultiNet2(TestCase):
    

    def test_init(self):
        datagen = DataGenMultiNet(decimalEncoder = encoderMultiNet2,lookback = 50)
        

    def test_get_data(self):
        datagen = DataGenMultiNet(decimalEncoder = encoderMultiNet2, batchSize = 10,  lookback=50)
        x, [yNotes, yTimes] = datagen[0]
        print(x.shape)
        print(yNotes.shape)
        print(yTimes.shape)




class TestDataGenEmbeddedMultiNet(TestCase):
    

    def test_init(self):
        datagen = DataGenEmbeddedMultiNet(decimalEncoder= encoderMultiNet2, lookback = 50)
        

    def test_get_data(self):
        datagen = DataGenEmbeddedMultiNet(decimalEncoder= encoderMultiNet2, batchSize = 30,  lookback=50)
        [xNotes, xTimes], [yNotes, yTimes] = datagen[0]
        
    def test_lengths(self):
        datagen = DataGenEmbeddedMultiNet(decimalEncoder= encoderMultiNet2, batchSize = 30,  lookback=50)
        [xNotes, xTimes], [yNotes, yTimes] = datagen[0]
        self.assertEqual(xNotes.shape, xTimes.shape)





        