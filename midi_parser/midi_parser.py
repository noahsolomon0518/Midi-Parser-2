from os import walk, path
from mido import MidiFile
from numpy import argmin
from itertools import chain, accumulate
from numpy import nanargmin
import numpy as np 
from mido import MetaMessage, MidiTrack
import math
import logging

"""
This module provides tools that are strictly used for parsing midis
NOT encoding. All the encoding happens in the decimal encoders and 
data generators.
"""
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setLevel("DEBUG")
streamHandler.setFormatter(formatter)
logger = logging.getLogger("MidiParser")
logger.setLevel("DEBUG")
logger.addHandler(streamHandler)


def parseToMidos(paths):
    midiObjs = []

    if(type(paths) != list):
        paths = [paths]

    for _path in paths:
        mf = MidiFile(_path, type=0)
        mf.tracks[0].insert(0,MetaMessage("track_name", name = _path))
        midiObjs.append(mf)

    return midiObjs


def findMidis(folder, r=True):
    paths = []
    if(".mid" in folder):
        paths.append(folder)
        return paths

    for (dirpath, _, filenames) in walk(folder):
        for file in filenames:
            if ".mid" in file:
                paths.append(path.join(dirpath, file))
        if not r:
            return paths
    return paths






class MidiParser:
    def __init__(self, noteRange, smallestTimeUnit, convertToC, timeMeasurement, folder = None, debugLevel = logging.ERROR):
        """
        Standard parsing of a collection of midis
            There are a couple ways to use this class:
            1. Directly input already initted MidiParser into DecimalEncoder.
            2. Call serialize function and input the folder path into Decimal Encoder.
            3. Pass parsed OneTracks into MidiParser

        Parameters
        ----------
        noteRange: tuple
            Tuple containing lowest note number and highest note number
        smallestTimeUnit: float
            The smallest time unit that will be captured. If it is 1/32 then the smallest unit of
            time is a 32nd note
        convertToC: bool
            If all pieces should be converted to C key signature. Currently if a piece has an unknown key
            it will not be parsed. This will change in the future where the key will be predicted
        timeMeasurement: str -> ["relative", "duration"]
            If time of a note should be measured relative to next note or the duration of a note
        folder: str
            Path of folder with midis
        debugLevel: str -> ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            Level of verbose, debug being highest, critical being lowest
        """
        logger.handlers[0].setLevel(debugLevel)
        self.logger = logger
        self.noteRange = noteRange
        self.smallestTimeUnit = smallestTimeUnit
        self.convertToC = convertToC

        self.midiPaths = []
        if folder != None:
            self.queueMidis(folder)
        

        

        

    def queueMidis(self, folder, r=True):
        """
        Queue up midis to be parsed. Pretty self explanatory

        Parameters
        ----------
        folder: str
            Folder at which midi paths will be extracted from
        r: bool
            Whether the midi paths should be recursively extracted        
        """
        midis = findMidis(folder, r)
        self.logger.debug("Queued {} midis".format(len(midis)))
        self.midiPaths.extend(midis)



    def parse(self):
        """
        Parses all queued midis to standard OneTrack

        Returns: List of parsed OneTracks
        """
        self.logger.info("Started parsing {} midis".format(len(self.midiPaths)))
        if(len(self.midiPaths)==0):
            self.logger.warning("No midis to be parsed")

        ots = []
        self.midos = parseToMidos(self.midiPaths)
        for _mido in self.midos:
            ot = OneTrack(_mido, self.noteRange, self.smallestTimeUnit, self.convertToC)
            if(ot.valid):
                ots.append(ot.track)
        self.logger.info("Successfully parsed {} midis".format(len(ots)))
        return ots


        
        

    def getTimeDistribution(self):
        """
        Allows you to see how the time units are distributed

        Returns: List of frequency tuples ordered
        """
        pass

    
    def serialize(self, folder):
        """
        Writes parsed one tracks to disk

        Parameters
        ----------
        folder: str
            folder at which OneTracks will be written to
        """

        pass


    @staticmethod
    def deSerialize(folder):
        """
        As name suggest deserializes pickled OneTracks

        Parameters
        ----------
        folder: str
            folder at which OneTracks will be written to
        """
        pass



#Time is assumed to be measured in ticks. A function can convert to seconds and time units depending on tpb, tempo
class Note:

    MAX_NOTE = 127
    MIN_NOTE = 0
    NOTES_IN_OCTAVE = 12

    def __init__(self, type, time, pitch = None, instrument = None, velocity = None):
        """
        Custom Note data structure 
            This object has many optional instance variables and functions which make
            parsing, encoding, and transformations a lot easier. The different forms
            of times are useful for the TimeHierachy

        Parameters
        ----------
        type: str -> ["rest", "note_on", "note_off"]
            The type of Note it is
        pitch: int
            Midi integer representation of note pitch
        time: int
            Amount of time associated with Note. What time actually means depends if it is a 
            RelativeNote or DurationNote
        instrument: int
            Integer representation of instrument
        velocity: int
            Midi velocity
        """
        if(velocity==0 and type=="note_on"):
            type = "note_off"
        self.type = type
        self.pitch = self._noneify(pitch)
        self.time = self._noneify(time)
        self.velocity = self._noneify(velocity)
        self.instrument = self._noneify(instrument)

    def transpose(self, n):
        """
        Transpose pitch by n halfsteps

        Parameters
        ----------
        n: int
            Read above
        """
        self.pitch += n
        self._checkNote()

    def transposeByOctave(self, n):
        """
        Transpose pitch by n octaves

        Parameters
        ----------
        n: int
            Read above
        """
        self.pitch += n*Note.NOTES_IN_OCTAVE
        self._checkNote()


    def _noneify(self, var):
        if(var!=None):
            return var
        return None

    def _checkNote(self):
        if(self.pitch>Note.MAX_NOTE):
            raise Exception("Note pitch exceeds max midi note of 127")
        if(self.pitch<Note.MIN_NOTE):
            raise Exception("Note pitch exceeds min midi note of 0")


class RelativeNote(Note):
    def __init__(self, type, time, pitch = None, instrument = None, velocity = None):
        """
        Note time is measured in dt (change in time in ticks)
        """
        super().__init__(type = type, time = time, pitch = pitch, instrument = instrument, velocity = velocity)
    
    def convertToDurational(self, time):
        """
        When you have a relative note and its absolute time this returns the relative note with the
        absolute time.

        Parameters
        ----------
        time: int
            time in absolute
        """
        return DurationalNote(self.type, time, pitch = self.pitch, instrument = self.instrument, velocity = self.velocity)
    

    def __str__(self):
        return str({
            "type": self.type,
            "time": self.time,
            "pitch": self.pitch
        })


class DurationalNote(Note):
    def __init__(self, type, time, pitch = None, instrument = None, velocity = None):
        """
        Note time is measured in duration (time note is played for in ticks)
        """
        super().__init__(self, type, time, pitch = pitch, instrument = instrument, velocity = velocity)
    
    def convertToRelative(self, time):
        return RelativeNote(self.type, time, pitch = self.pitch, instrument = self.instrument, velocity = self.velocity)
    
    def __str__(self):
        return {
            "type": self.type,
            "time": self.time,
            "pitch": self.pitch
        }

#Implementation is not general. Specifically for the OneTracks.
class OneTrack:
    def __init__(self, mido, noteRange, smallestTimeUnit, convertToC):
        """
        OneTracks automatically perform many functions that decimal encoders need in order to do their encoding.
        As the name suggest the tracks of a midi are flattened to one. 
        """
        self.convertToC = convertToC
        self.name = mido.tracks[0][0].name
        self.key = self._extractKeySignature(mido)
        self.tpb = mido.ticks_per_beat
        self.smallestTimeUnit = smallestTimeUnit
        self.channels = []
        self.currentTime = 0
        self.track = None
        self.valid = self._checkValid()
        if(self.valid):
            for track in mido.tracks:
                self._addChannel(track)
            self.track = self._flatten()
        else:
            logger.debug("No key signature found for {}".format(self.name))
    

    

    def _extractKeySignature(self, mido):
        for track in mido.tracks:
            for msg in track:
                if(type(msg)==MetaMessage and msg.type == "key_signature"):
                    return msg.key
        return None

    def _flatten(self):
        notes = []
        for i in range(sum([len(channel)-1 for channel in self.channels])):
            increasedTimes = [channel.getNextIncreasedTime() for channel in self.channels]
            ind = self._getArgMin(increasedTimes)
            nextNote, noteTime = self.channels[ind].advance()
            nextNote.time = noteTime - self.currentTime
            self.currentTime += nextNote.time
            if(self._isNote(nextNote)):
                relativeNote = RelativeNote(type = nextNote.type, time = self._timeConversion(nextNote.time), pitch = nextNote.note, velocity = nextNote.velocity)
                notes.append(relativeNote)
        return notes


    def _checkValid(self):
        if(self.convertToC == True and self.key == None):
            return False
        return True


    def _addChannel(self,lst):
        self.channels.append(Channel(lst))

    def _timeConversion(self, _time):
        return int(round(_time*(1/self.tpb)/4/self.smallestTimeUnit))

    def _isNote(self,msg):
        return (type(msg) != MetaMessage and msg.type in ["note_on", "note_off"])

    def _getArgMin(self, increasedTimes):
        return nanargmin(increasedTimes)
    
    def _advanceChannel(self, ind):
        self.channels[ind].advance()



class Channel:
    def __init__(self, lst):
        self.data = lst
        self.curInd = 0
        self.curDt = lst[0].time
        self.curTime = 0
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def getNextIncreasedTime(self):
        return self.curDt + self.curTime if self.curInd != len(self) - 1 else np.NaN

    def advance(self):
        self.curInd += 1
        self.curTime += self.curDt
        self.curDt = self.data[self.curInd].time
        return self.data[self.curInd-1], self.curTime




    



    
    
        

        








