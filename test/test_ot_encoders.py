
from unittest import TestCase
import unittest
from midi_parser.one_tracks import OneTrack, OneTrackOnOnly, Note
from midi_parser.ot_encoders import OTEncoderOnOff, OTEncoderMultiNet
from midi_parser.decimal_encoders import parseToMidos, findMidis


midos = parseToMidos(findMidis("test/test_data/midis"))
ots = [OneTrack(mido, 1/32) for mido in midos]
otsOnOnly = [OneTrackOnOnly(mido, smallestTimeUnit = 1/32, convertToC=True) for mido in midos]

class TestOTEncoderOnOff(TestCase):


    def test_init(self):
        otsEncoded = OTEncoderOnOff(ots, 40).encodedOTs
        self.assertGreater(len(otsEncoded), 0)
        for otEncoded in otsEncoded:
            self.assertGreater(len(otEncoded), 0)




class TestOTEncoderMultiNet(TestCase):


    def test_init(self):
        otsEncoded = OTEncoderMultiNet(otsOnOnly, 40).encodedOTs
        print(otsEncoded)
        self.assertGreater(len(otsEncoded), 0)
        for otEncoded in otsEncoded:
            self.assertEqual(len(otEncoded), 2)


