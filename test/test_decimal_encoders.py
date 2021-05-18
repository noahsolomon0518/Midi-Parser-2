from midi_parser.midi_parser import MidiParser
from unittest import TestCase
import unittest
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet, DecimalEncoderMultiNet2

class TestDecimalEncoderOnOff(TestCase):

    
    def test_init(self):
        
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_relative")
        encoder = DecimalEncoderOnOff(parsedMidis)
    

    def test_encode(self):
        mp = MidiParser((46, 1/32, True, "relative", "test/test_data/midis/Bwv768 Chorale and Variations/what"))
        parsed = mp.parse()
        encoder = DecimalEncoderOnOff(parsed)
        encoded = encoder.encode()
        for piece in encoded:
            self.assertGreater(len(piece), 0)
            print(piece)



    



class TestOTEncoderMultiNet(TestCase):

    
    def test_init(self):
        encoder = DecimalEncoderMultiNet("test/test_data/midis", 1/32, 30)
        data = encoder.encode()
    

    def test_encoder(self):
        encoder = DecimalEncoderMultiNet("test/test_data/midis", 1/32, 30)
        data = encoder.encode()
        print(data)
        for piece in data:
            self.assertEqual(len(piece), 2)
            self.assertGreater(len(piece[0]), 0)
            self.assertGreater(len(piece[1]), 0)





class TestOTEncoderMultiNet2(TestCase):

    
    def test_init(self):
        encoder = DecimalEncoderMultiNet2("test/test_data/midis", 1/32, 30)
        data = encoder.encode()
    

    def test_encoder(self):
        encoder = DecimalEncoderMultiNet2("test/test_data/midis", 1/32, 30)
        data = encoder.encode()
        #print(data)
        for piece in data:
            self.assertEqual(len(piece), 2)
            self.assertGreater(len(piece[0]), 0)
            self.assertGreater(len(piece[1]), 0)