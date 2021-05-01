from unittest import TestCase
import unittest
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet, DecimalEncoderMultiNet2

class TestOTEncoderOnOff(TestCase):

    
    def test_init(self):
        encoder = DecimalEncoderOnOff("test/test_data/midis", 1/32, 30)


    def test_encode(self):
        encoder = DecimalEncoderOnOff("test/test_data/midis", 1/32, 30)
        data = encoder.encode()
        for piece in data:
            self.assertGreater(len(piece), 0)



    



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