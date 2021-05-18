import itertools
from os import walk, path
from mido import MidiFile
from numpy import argmin
from itertools import chain, accumulate
from numpy import nanargmin
import numpy as np 
from mido import MetaMessage, MidiTrack
import math
import logging
import pickle

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
        General use parser for midis

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
        self.timeMeasurement = timeMeasurement
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
            ot = OneTrack(_mido, self.noteRange, self.smallestTimeUnit, self.convertToC, self.timeMeasurement)
            if(ot.valid):
                ots.append(ot.track)
        self.logger.info("Successfully parsed {} midis".format(len(ots)))
        return ots



    
    def serialize(self, fp):
        """
        Writes parsed one tracks to disk

        Parameters
        ----------
        fp: str
            folder at which OneTracks will be written to
        """
        parsed = self.parse()
        f = open(fp,"wb")
        pickle.dump(parsed,f)
        f.close()
        logger.info("Successfully serialized parsed midis at {}".format(fp))
        


    @staticmethod
    def deSerialize(fp):
        """
        As name suggest deserializes pickled OneTracks

        Parameters
        ----------
        fp: str
            folder at which OneTracks will be written to
        """
        f = open(fp,"rb")
        parsed = pickle.load(f)
        f.close()
        return parsed



#Time is assumed to be measured in ticks. A function can convert to seconds and time units depending on tpb
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
        type: str -> ["time_unit", "note_on", "note_off"]
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

    def convertToRelative(self, time):
        """
        Parameters
        ----------
        time: int
            time in relative dt
        """
        return RelativeNote(self.type, time, pitch = self.pitch, instrument = self.instrument, velocity = self.velocity)
    

    def convertToDurational(self, time):
        """
        Parameters
        ----------
        time: int
            time in absolute
        """
        return DurationalNote(self.type, time, pitch = self.pitch, instrument = self.instrument, velocity = self.velocity)
    
    def convertToAbsolute(self, time):
        """
        Parameters
        ----------
        time: int
            time in absolute
        """
        return AbsoluteNote(self.type, time, pitch = self.pitch, instrument = self.instrument, velocity = self.velocity)


    def _noneify(self, var):
        if(var!=None):
            return var
        return None

    def _checkNote(self):
        if(self.pitch>Note.MAX_NOTE):
            raise Exception("Note pitch exceeds max midi note of 127")
        if(self.pitch<Note.MIN_NOTE):
            raise Exception("Note pitch exceeds min midi note of 0")
    
    def __str__(self):
        return str({
            "type": self.type,
            "time": self.time,
            "pitch": self.pitch
        })


class AbsoluteNote(Note):
    def __init__(self, type, time, pitch = None, instrument = None, velocity = None):
        """
        Note time is measure in absolute ticks
        """
        super().__init__(type = type, time = time, pitch = pitch, instrument = instrument, velocity = velocity)

class RelativeNote(Note):
    def __init__(self, type, time, pitch = None, instrument = None, velocity = None):
        """
        Note time is measured in dt (change in time in ticks)
        """
        super().__init__(type = type, time = time, pitch = pitch, instrument = instrument, velocity = velocity)

class DurationalNote(Note):
    def __init__(self, type, time, pitch = None, instrument = None, velocity = None):
        """
        Note time is measured in duration (time note is played for in ticks)
        """
        super().__init__(type, time, pitch = pitch, instrument = instrument, velocity = velocity)
    

#Implementation is not general. Specifically for the OneTracks.
class OneTrack:

    halfStepsAboveC = {
        "C":0,
        "B#":0,
        "Db":1,
        "C#":1,
        "D":2,
        "Eb":3,
        "D#":3,
        "Fb":4,
        "E":4,
        "E#":5,
        "F":5,
        "F#":6,
        "Gb":6,
        "G":7,
        "G#":8,
        "Ab":8,
        "A":9,
        "A#":10,
        "Bb":10,
        "B":11
    }
    
    def __init__(self, mido, noteRange, smallestTimeUnit, convertToC, timeMeasurement, name = None):
        """
        OneTracks automatically perform many functions that decimal encoders need in order to do their encoding.
        As the name suggest the tracks of a midi are flattened to one. 
        """
        self.minNote, self.maxNote = noteRange
        self.convertToC = convertToC
        try:
            self.name = mido.tracks[0][0].name
        except: 
            self.name = name
        self.key = self._extractKeySignature(mido)
        self.tpb = mido.ticks_per_beat
        self.smallestTimeUnit = smallestTimeUnit
        self.channels = []
        self.currentTime = 0
        self.track = None
        self.valid = self._checkValid()
        if(self.valid):
            self.halfStepsBelowC = 12 - OneTrack.halfStepsAboveC[self.key.replace("m", "")]
            self.track = self._flatten(mido)
            if(len(self.track) < 10):      
                self.valid == False
                logger.debug("Piece {} has length of 0".format(self.name))
            if(timeMeasurement=="durational"):
                self._convertDurational()
            self._applyConstraints()
        else:
            logger.debug("No key signature found for {}".format(self.name))
        
    




    def _extractKeySignature(self, mido):
        for track in mido.tracks:
            for msg in track:
                if(type(msg)==MetaMessage and msg.type == "key_signature"):
                    return msg.key
        return None

    #Combines all synchronous tracks together into one track
    def _flatten(self, mido):
        notes = []

        absoluteTrack = list(itertools.chain.from_iterable([self._convertAbsolute(track) for track in mido.tracks]))
        absoluteTrack.sort(key = lambda x: x.time)

        relativeTrack = self._convertRelative( absoluteTrack)
        return relativeTrack
    

    def _convertAbsolute(self, track):
        absoluteTime = 0
        absoluteTrack = []
        for msg in track:
            absoluteTime += msg.time
            if(self._isNote(msg)):
                absoluteTrack.append(AbsoluteNote(msg.type, absoluteTime, msg.note, velocity=msg.velocity))
        return absoluteTrack

    
    def _convertRelative(self, absoluteTrack):
        
        return [absoluteTrack[i].convertToRelative(absoluteTrack[i+1].time - absoluteTrack[i].time) for i in range(len(absoluteTrack)-1)] + [absoluteTrack[-1].convertToRelative(0)]
    




    def _applyConstraints(self):
        if(self.convertToC):
            for note in self.track:
                self._applyNoteRange(note)
                self._convertToC(note)
                self._timeConversion(note)
        else:
            for note in self.track:
                self._applyNoteRange(note)
                self._timeConversion(note)


            


    
        
    #If timeMeasurement is durational 
    def _convertDurational(self):
        """
        Note times are durational.
        """
        notesTimed = []
        for i, note in enumerate(self.track):
            if(note.time>0):
                notesTimed.append(DurationalNote("time_unit", note.time))
            if(note.type == "note_on"):
                noteNum = note.pitch
                dt = 0
                for nextNote in self.track[i:]:
                    if(nextNote.type == "note_off" and nextNote.pitch == noteNum):
                        dt+=nextNote.time 
                        break
                    dt+=nextNote.time 
                notesTimed.append(note.convertToDurational(dt))
        self.track = notesTimed



    def _checkValid(self):
        if(self.convertToC == True and self.key == None):
            return False
        return True

    def _convertToC(self, note):
        note.transpose(self.halfStepsBelowC if self.halfStepsBelowC<=6 else self.halfStepsBelowC - 12)
        
    def _applyNoteRange(self, note):
        if(note.pitch>self.maxNote):
            octavesToShift = (((note.pitch-self.maxNote)//12) + 1)
            note.trasnpose(-12 * octavesToShift)
        elif(note.pitch<self.minNote):
            octavesToShift = (((self.minNote - note.pitch)//12) + 1)
            note.transpose(12 * octavesToShift)
    
    #converts ticks to number of time units
    def _timeConversion(self, note):
        converted = note.time*(1/self.tpb)/4/self.smallestTimeUnit
        if(converted>0 and converted<1):
            note.time = 1
        note.time = round(converted)


    def _isNote(self,msg):
        return (type(msg) != MetaMessage and msg.type in ["note_on", "note_off"])











class MidiAnalyzer:

    def __init__(self, folder):
        """
        Used for configuring parameters in MidiParser. For example smallest time unit

        Parameters
        ----------
        folder: str
            folder of midis
        """

        self.midos = parseToMidos(findMidis(folder))





