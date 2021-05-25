from midi_parser.midi_parser import MidiParser
from unittest import TestCase
import numpy
from midi_parser.pieces import OnOffPiece, MultiNetPiece
from midi_parser.data_generators import DataGenGuideNet, DataGenOnOff, DataGenMultiNet, DataGenEmbeddedMultiNet, DataGen
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet

relativeParsed = MidiParser.deSerialize("test/test_data/serialized_relative")
relativeParsed = MidiParser((46,84), 1/128, True, "relative", folder = "test/test_data/midis").parse()
encoded = DecimalEncoderOnOff(relativeParsed).encode()
durParsed = MidiParser((46,84), 1/128, True, "durational", folder = "test/test_data/midis").parse()
encodedMultiNet = DecimalEncoderMultiNet(durParsed).encode()


class TestDataGen(TestCase):

    def test_init(self):
        datagen = DataGen(encoded, 32, 100, 5)
        #datagen.__getitem__(0)

class TestDataGeneratorOnOff(TestCase):
    

    def test_init(self):
        datagen = DataGenOnOff(encoded,32, 100, 5)
        

    def test_get_data(self):
        datagen = DataGenOnOff(encoded, 32, 100, 5)
        data = datagen.__getitem__(0)
        print(data)

    def test_play_sample(self):
        mp = MidiParser((46, 84), 1/64, True, "relative", folder = "test/test_data/midis/Bwv768 Chorale and Variations", debugLevel="DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderOnOff(parsed)
        encoded = encoder.encode()
        datagen = DataGenOnOff(encoded, 32, 100, 5)
        data = datagen.__getitem__(5)
        piece = OnOffPiece(datagen.ohe.inverse_transform(data[0][0]).reshape(-1).tolist(), 1/64)
        piece.play()


import itertools
class TestDataGeneratorMultiNet(TestCase):
    
    def test_init(self):
        datagen = DataGenMultiNet(encodedMultiNet, 32, 100, 5)
        

    def test_get_data(self):
        datagen = DataGenMultiNet(encodedMultiNet, 32, 100, 5)
        data = datagen.__getitem__(0)

    def test_play_sample(self):
        mp = MidiParser((46, 84), 1/32, True, "durational", folder="test/test_data/midis/Bwv768 Chorale and Variations", debugLevel="DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderMultiNet(parsed)
        encoded = encoder.encode()[:2]
        datagen = DataGenMultiNet(encoded, 32, 100, 5)
        data = datagen.__getitem__(0)
        piece = MultiNetPiece(datagen.ohe.inverse_transform(data[0][0]), 1/32)
        piece.play()







import numpy as np
class TestDataGenEmbeddedMultiNet(TestCase):
    

    def test_init(self):
        datagen = DataGenEmbeddedMultiNet(encodedMultiNet, 32, 100, 5)
        

    def test_get_data(self):
        datagen = DataGenEmbeddedMultiNet(encodedMultiNet, 32, 100, 5)
        [xNotes, xTimes], [yNotes, yTimes] = datagen[0]
        
    def test_play_sample(self):
        mp = MidiParser((46, 84), 1/128, True, "durational", folder="test/test_data/midis", debugLevel="DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderMultiNet(parsed)
        encoded = encoder.encode()
        datagen = DataGenEmbeddedMultiNet(encoded, 32, 100, 5)
        data = datagen.__getitem__(5)
        notes = data[0][0][0]
        times = data[0][1][0]
        piece = np.stack([datagen.ordEnc.inverse_transform([(notes[i], times[i])]) for i in range(len(notes))]).reshape(-1,2)
        piece = MultiNetPiece(piece, 1/128)
        piece.play()


class TestDataGenGuideNet(TestCase):
    def test_init(self):
        datagen = DataGenGuideNet(encodedMultiNet, 32, 100, 5)
        

    def test_get_data(self):
        datagen = DataGenGuideNet(encodedMultiNet, 32, 100, 5)
        [xNotes, xTimes], [yNotes, yTimes] = datagen[0]
        print(xNotes,xTimes,yNotes,yTimes)

    def test_play(self):
        datagen = DataGenGuideNet(encodedMultiNet, 32, 100, 5)
        data = datagen.__getitem__(5)
        notes = data[0][0][0]
        times = data[0][1][0]
        piece = np.stack([datagen.ordEnc.inverse_transform([(notes[i], times[i])]) for i in range(len(notes))]).reshape(-1,2)
        piece = MultiNetPiece(piece, 1/128)
        piece.play()





        