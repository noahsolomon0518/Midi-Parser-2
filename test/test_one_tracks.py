from unittest import TestCase
import unittest
from midi_parser.one_tracks import OneTrack, OneTrackOnOnly, OneTrackOnOnlyOrdered, Note
from midi_parser.decimal_encoders import parseToMidos, findMidis


midos = parseToMidos(findMidis("test/test_data/midis"))



class TestOneTrackAbstraction(TestCase):


    def test_init(self):
        ot = OneTrack(midos[4], 1/32)
        self.assertGreater(len(ot.notesRel), 0)
        self.assertIsInstance(ot.notesRel[0], Note)



    def test_range(self):
        ot = OneTrack(midos[4], 1/32, noteRange=(65,100))
        for note in ot.notesRel:
            self.assertGreaterEqual(note.note,60)


class TestOneTrackOnOnly(TestCase):


    def test_init(self):
        ot = OneTrackOnOnly(midos[4], 1/64)
        self.assertGreater(len(ot.notesRel), 0)
        self.assertIsInstance(ot.notesRel[0], Note)



    def test_range(self):
        ot = OneTrack(midos[4], 1/32,noteRange= (65,100))
        for note in ot.notesRel:
            self.assertGreaterEqual(note.note,60)



class TestOneTrackOnOnlyOrdered(TestCase):


    def test_init(self):
        ot = OneTrackOnOnlyOrdered(midos[4], 1/64, noteRange=(65,100))
        self.assertGreater(len(ot.notesRel), 0)
        self.assertIsInstance(ot.notesRel[0], Note)





    def test_range(self):
        ot = OneTrackOnOnlyOrdered(midos[4], 1/32, noteRange=(65,100))
        for note in ot.notesRel:
            self.assertGreaterEqual(note.note,60)
    

    def test_ordered(self):
        ot = OneTrackOnOnlyOrdered(midos[4], 1/32, noteRange=(65,100))
        for i,note in enumerate(ot.notesTimed):
            if(note.note<300 and i<(len(ot.notesTimed)-1)):
                self.assertLess(note.note, ot.notesTimed[i+1].note)
         


