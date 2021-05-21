import struct
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


    def test_parse_relative(self):
        mp = MidiParser((46,84), 1/32, True, "relative", mode = "major", folder = "test/test_data/midis", debugLevel = "DEBUG")
        parsed = mp.parse()
        self.assertGreater(len(parsed),0)

    

    def test_parse_durational(self):
        mp = MidiParser((46,84), 1/32, True, "durational", folder = "test/test_data/midis", debugLevel = "DEBUG")
        parsed = mp.parse()
        self.assertGreater(len(parsed),0)
        for note in parsed[0]:
            print(note)
    def test_serialize(self):
        mp = MidiParser((46,84), 1/32, True, "relative", folder = "test/test_data/midis", debugLevel = "DEBUG")
        mp.serialize("test/test_data/serialized_relative")
    
    def test_deserialize(self):
        parsed = MidiParser.deSerialize("test/test_data/serialized_durational")

    def test_play(self):

        parsed = MidiParser.deSerialize("test/test_data/serialized_relative")


        



class TestOneTrack(TestCase):



    def test_with_single_midi(self):
        mf = MidiFile("test/test_data/midis/Bwv768 Chorale and Variations/bsgjg_a.mid")
        ot = OneTrack(mf, (46,84), 1/32, True, "relative")
        print("ticks per beats",ot.tpb)



        
    def test_play(self):
        
        stu = 1/128
        timeUnitSeconds = 2*stu
        mf = MidiFile("test/test_data/midis/Bwv768 Chorale and Variations/what/bsgjg_e.mid")
        ot = OneTrack(mf,  (22,96), stu, True, "relative")
        print(ot.tpb)
        fs = fluidsynth.Synth()
        fs.start()
        sfid = fs.sfload(sf2)
        fs.program_select(0, sfid, 0, 0)
        for msg in ot.track:
            print(msg.time)
            if(msg.type == "note_on"):
                fs.noteon(0, msg.pitch, 100)
            if(msg.type == "note_off"):
                fs.noteoff(0, msg.pitch)

            if(msg.time>0):
                time.sleep((msg.time)*timeUnitSeconds)
  
          


