from midi_parser.midi_parser import MidiParser
from unittest import TestCase

from midi_parser.pieces import OnOffPiece, MultiNetPiece
from midi_parser.data_generators import DataGenOnOff, DataGenMultiNet, DataGenEmbeddedMultiNet
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet

relativeParsed = MidiParser.deSerialize("test/test_data/serialized_relative")
relativeParsed = MidiParser((46,84), 1/128, True, "relative", "test/test_data/midis").parse()
encoded = DecimalEncoderOnOff(relativeParsed).encode()
durParsed = MidiParser((46,84), 1/128, True, "durational", "test/test_data/midis").parse()
encodedMultiNet = DecimalEncoderMultiNet(durParsed).encode()


class TestDataGeneratorOnOff(TestCase):
    

    def test_init(self):
        datagen = DataGenOnOff(encoded,50)
        

    def test_get_data(self):
        datagen = DataGenOnOff(encoded, 10,  lookback=50)
        data = datagen.__getitem__(0)
        print(data)

    def test_play_sample(self):
        mp = MidiParser((46, 84), 1/64, True, "relative", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderOnOff(parsed)
        encoded = encoder.encode()
        datagen = DataGenOnOff(encoded, 32, 100, 5)
        data = datagen.__getitem__(5)
        print(encoded[0])
        piece = OnOffPiece(datagen.ohe.inverse_transform(data[0][0]).reshape(-1).tolist(), 1/64)
        piece.play()



class TestDataGeneratorMultiNet(TestCase):
    
    def test_init(self):
        datagen = DataGenMultiNet(encodedMultiNet,50)
        

    def test_get_data(self):
        datagen = DataGenMultiNet(encodedMultiNet, 10,  lookback=50)
        data = datagen.__getitem__(0)

    def test_play_sample(self):
        mp = MidiParser((46, 84), 1/64, True, "durational", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderMultiNet(parsed)
        encoded = encoder.encode()
        datagen = DataGenMultiNet(encoded, 32, 100, 5)
        data = datagen.__getitem__(5)
        piece = MultiNetPiece(datagen.ohe.inverse_transform(data[0][0]).reshape(-1).tolist(), 1/64)
        piece.play()








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





        