from midi_parser.midi_parser import MidiParser
from unittest import TestCase
import unittest
from midi_parser.decimal_encoders import DecimalEncoderMiniBachStyle, DecimalEncoderOnOff, DecimalEncoderMultiNet
import itertools
from midi_parser.pieces import MultiNetPiece, OnOffPiece



    

class TestDecimalEncoderOnOff(TestCase):


    def test_init(self):
        
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_relative")
        encoder = DecimalEncoderOnOff(parsedMidis, 100)
    

    def test_encode(self):
        mp = MidiParser((46, 84), 1/32, True, "relative", folder = "test/test_data/midis/Bwv768 Chorale and Variations", debugLevel="DEBUG")
        parsed = mp.parse()
        print(parsed)
        encoder = DecimalEncoderOnOff(parsed,  100)
        encoded = encoder.encode()
        for piece in encoded:
            self.assertGreater(len(piece), 0)
            for note in piece:
                self.assertGreater(350, note)

    def test_play(self):
        mp = MidiParser((46, 84), 1/128, True, "relative", folder = "test/test_data/midis/Bwv768 Chorale and Variations", debugLevel="DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderOnOff(parsed,  100)
        encoded = encoder.encode()
        piece = OnOffPiece(encoded[0], 1/128)
        piece.play()


    

import numpy as np

class TestDecimalEncoderMultiNet(TestCase):
    
    def test_init(self):
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_durational")
        encoder = DecimalEncoderMultiNet(parsedMidis, 100)
    

    def test_encoder(self):
        parsedMidis = MidiParser.deSerialize("test/test_data/serialized_durational")
        encoder = DecimalEncoderMultiNet(parsedMidis, 100)
        print(encoder.encode()[0])


    def test_play(self):
        mp = MidiParser((46, 84), 1/32, True, "durational", "both", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        parsed = mp.parse()
        encoder = DecimalEncoderMultiNet(parsed, 100)
       
        piece = encoder.encode()[0]
        piece = MultiNetPiece(piece, 1/32)
        piece.play()
        

class TestDecimalEncoderMiniBachStyle(TestCase):
    
    def test_init(self):
        mp = MidiParser((46, 84), 1/128, True, "by_time_unit", "both", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        encoder = DecimalEncoderMiniBachStyle(mp.parse())
    

    def test_encoder(self):
        mp = MidiParser((46, 84), 1/128, True, "by_time_unit", "both", "test/test_data/midis/Bwv768 Chorale and Variations", "DEBUG")
        encoder = DecimalEncoderMiniBachStyle(mp.parse())
        print(encoder.encode()[0])



        


