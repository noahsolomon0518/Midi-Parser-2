from unittest import TestCase
from midi_parser.midi_parser import MidiParser,  OneTrack
import mido
from mido import MidiFile

import fluidsynth
import time
import os
import numpy as np
from mido import MidiFile, MidiTrack, Message
sf2 = os.path.abspath("C:/Users/noahs/Local Python Libraries/soundfonts/piano.sf2")


class TestMidiParser(TestCase):


    def test_init(self):
        mp = MidiParser((46,84), 1/32, True, "relative", folder = "test/test_data/midis")


    def test_paths(self):
        mp = MidiParser((46,84), 1/32, True, "relative", folder = "test/test_data/midis", debugLevel = "DEBUG")
        self.assertGreater(len(mp.midiPaths), 0)


    def test_parse(self):
        mp = MidiParser((46,84), 1/32, True, "relative", folder = "test/test_data/midis", debugLevel = "DEBUG")
        parsed = mp.parse()
        self.assertGreater(len(parsed),0)
        



class TestOneTrack(TestCase):



    def test_with_single_midi(self):
        mf = MidiFile("test/test_data/midis/Bwv768 Chorale and Variations/bsgjg_a.mid")
        ot = OneTrack(mf, 0, 1/32)
        print("ticks per beats",ot.tpb)
 


        
    def test_play(self):
        timeUnitSeconds = 4/128
        mf = MidiFile("test/test_data/midis/Bwv768 Chorale and Variations/bsgjg_a.mid")
        ot = OneTrack(mf, 0, 1/64)
        fs = fluidsynth.Synth()
        fs.start()
        sfid = fs.sfload(sf2)
        fs.program_select(0, sfid, 0, 0)
        for msg in ot.track:
            print(msg.time)
            if(msg.time>0):
                time.sleep((msg.time)*timeUnitSeconds)
            if(msg.type == "note_on"):
                fs.noteon(0, msg.pitch, 100)
            else:
                fs.noteoff(0, msg.pitch)
