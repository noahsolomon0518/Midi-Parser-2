from midi_parser.midi_parser import MidiParser
from unittest import TestCase
import unittest
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet
import itertools
from midi_parser.pieces import MultiNetPiece

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



    

import numpy as np

class TestDecimalEncoderMultiNet(TestCase):

    
    def test_init(self):
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_durational")
        encoder = DecimalEncoderMultiNet(parsedMidis)
    

    def test_encoder(self):
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_durational")
        encoder = DecimalEncoderMultiNet(parsedMidis)
        print(encoder.encode()[0])


    def test_play(self):
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_durational")
        encoder = DecimalEncoderMultiNet(parsedMidis)
        piece = list(itertools.chain.from_iterable(encoder.encode()[0]))
        piece = MultiNetPiece(piece, 1/64)
        piece.play()
        


