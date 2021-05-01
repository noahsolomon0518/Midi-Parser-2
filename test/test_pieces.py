from unittest import TestCase
import unittest
from midi_parser.music_generators import OnOffPiece, MultiNetPiece
from midi_parser.decimal_encoders import DecimalEncoderOnOff, DecimalEncoderMultiNet, DecimalEncoderMultiNet2

encoded = DecimalEncoderOnOff("test/test_data/midis", 1/64, 50).encode()
encodedMulti = DecimalEncoderMultiNet("test/test_data/midis", 1/64, 50).encode()
encodedMulti2 = DecimalEncoderMultiNet2("test/test_data/midis", 1/64, 50).encode()

class TestOnOffPiece(TestCase):


    def test_init(self):
        piece = OnOffPiece(encoded[0], 1/64)
        self.assertGreater(len(piece.piece), 0)


    def test_play(self):
        piece = OnOffPiece(encoded[0][:80], 1/64)
        piece.play()

    def test_save(self):
        piece = OnOffPiece(encoded[0][:], 1/64)
        piece.save("test/test_data/on_off_piece_test.mid")

class TestMultiNetPiece(TestCase):


    def test_init(self):
        piece = MultiNetPiece(encodedMulti[0], 1/64)
        self.assertGreater(len(piece.piece), 0)


    def test_play(self):
        piece = MultiNetPiece([encodedMulti[0][0][:50],encodedMulti[0][1][:50]] , 1/64)
        piece.play()
        



#Same as multinet piece just slightly different algoritm that orders notes in a time unit
class TestMultiNet2Piece(TestCase):


    def test_init(self):
        piece = MultiNetPiece(encodedMulti2[0], 1/64)
        self.assertGreater(len(piece.piece), 0)


    def test_play(self):
        piece = MultiNetPiece([encodedMulti2[0][0][:50],encodedMulti2[0][1][:50]] , 1/64)
        piece.play()

