#Encodes all queued midis to list of integers

from midi_parser.midi_parser import RelativeNote
from mido import MidiFile
from .one_tracks import *

from os import walk
from os import path
import math
import warnings


class DecimalEncoder:
    def __init__(self, parsedMidis, nClassesTimes):
        """
        Abstract class for all implementations of decimal encoders

        Parameters
        ----------

        """
        self.parsedMidis = parsedMidis

    def encode(self):
        return [self._encodeOne(piece) for piece in self.parsedMidis]

    #Abstract function that decimal encoders must include.
    def _encodeOne(self, piece):
        raise NotImplementedError("Add encode() function")





class DecimalEncoderOnOff(DecimalEncoder):
    def __init__(self, parsedMidis):
        """
        note_ons -> <relative note> + 150
        note_offs -> <relative note>
        time_unit -> 299 + <time>
        """
        try:
            assert type(parsedMidis[0][0]) == RelativeNote
        except AssertionError:
            raise AssertionError("Parsed midis must have relative notes. Use timeMeasurement = \"relative\" in MidiParser init") 
        super().__init__(parsedMidis)
    

    def _encodeOne(self, piece):
        oneEncoded = []
        for note in piece:
            if(note.type == "note_on"):
                oneEncoded.append(150+note.pitch)
            else:
                oneEncoded.append(note.pitch)
            if(note.time>0):
                oneEncoded.append(299+note.time)
        return self._order(oneEncoded)
    
    def _order(self, oneEncoded):
        pointer = 0
        ordered = []
        while pointer<len(oneEncoded):
            curStream = []
            while pointer<len(oneEncoded) and oneEncoded[pointer]<300:
                curStream.append(oneEncoded[pointer])
                pointer+=1
            if(pointer<len(oneEncoded)):
                curStream.append(oneEncoded[pointer])
            ordered.extend(sorted(list(set(curStream))))
            pointer+=1
        return ordered











        





class DecimalEncoderMultiNet(DecimalEncoder):
    def __init__(self, folder, smallestTimeUnit, nClassesTimes, noteRange = (36, 84) ,debug=False, r=True, convertToC = True,  scales = "both"):
        
        super().__init__(
            folder, 
            smallestTimeUnit = smallestTimeUnit,
            nClassesTimes = nClassesTimes,
            noteRange = noteRange,
            debug=debug, 
            r=r, 
            convertToC = convertToC,  
            scales = scales)
    

    #Uses vanilla OneTrack to encode
    def _initOneTrack(self, mido):
        return OneTrackOnOnly(
            mido, 
            convertToC = self.convertToC, 
            noteRange = (self.minNote, self.maxNote),
            scales = self.scales, 
            smallestTimeUnit = self.smallestTimeUnit)

    def _initOTEncoder(self, oneTracks):
        return OTEncoderMultiNet(oneTracks, nClassesTimes = self.nClassesTimes)




class DecimalEncoderMultiNet2(DecimalEncoder):
    """
    Slightly different implementation of multinet 1.
    Orders notes within each timestep and deletes duplicates

    Used with DataGenMultiNet and GeneratorMultiNet



    """
    def __init__(self, folder, smallestTimeUnit, nClassesTimes, noteRange = (36, 84) ,debug=False, r=True, convertToC = True,  scales = "both"):
        
        super().__init__(
            folder, 
            smallestTimeUnit = smallestTimeUnit,
            nClassesTimes = nClassesTimes,
            noteRange = noteRange,
            debug=debug, 
            r=r, 
            convertToC = convertToC,  
            scales = scales)
    

    def _initOneTrack(self, mido):
        return OneTrackOnOnlyOrdered(
            mido, 
            convertToC = self.convertToC, 
            noteRange = (self.minNote, self.maxNote),
            scales = self.scales, 
            smallestTimeUnit = self.smallestTimeUnit)

    def _initOTEncoder(self, oneTracks):
        return OTEncoderMultiNet(oneTracks, nClassesTimes = self.nClassesTimes)












