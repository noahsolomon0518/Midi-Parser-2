from midi_parser.midi_parser import MidiParser
from unittest import TestCase
import unittest
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet
import itertools
from midi_parser.pieces import MultiNetPiece, OnOffPiece

class TestDecimalEncoderOnOff(TestCase):

    
    def test_init(self):
        
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_relative")
        encoder = DecimalEncoderOnOff(parsedMidis)
    

    def test_encode(self):
        mp = MidiParser((46, 84), 1/32, True, "relative", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderOnOff(parsed)
        encoded = encoder.encode()
        for piece in encoded:
            self.assertGreater(len(piece), 0)
            print(piece)

    def test_play(self):
        mp = MidiParser((46, 84), 1/128, True, "relative", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderOnOff(parsed)
        encoded = encoder.encode()
        piece = OnOffPiece(encoded[0], 1/128)
        piece.play()


    

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
        mp = MidiParser((46, 84), 1/32, True, "durational", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderMultiNet(parsed)
       
        piece = encoder.encode()[0]
        piece = MultiNetPiece(piece, 1/32)
        piece.play()
        


