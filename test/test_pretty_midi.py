from unittest import TestCase
import pretty_midi
from midi_parser.midi_parser import findMidis
class TestPrettyMidi(TestCase):

    def test(self):
        paths = findMidis("test")
        midis = []
        for midi in paths:
            midis.append(pretty_midi.PrettyMIDI(midi))
        print(midis[0].get_beats())


