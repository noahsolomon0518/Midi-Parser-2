#All the different forms of pieces that can be generated

import fluidsynth
import time
import os
import numpy as np
from mido import MidiFile, MidiTrack, Message
sf2 = os.path.abspath("C:/Users/noahs/Local Python Libraries/soundfonts/piano.sf2")


TIME_UNIT_START = 300
N_POSSIBLE_NOTES = 150




class Player:
    """
    Abstract class for piece
    Has a function that converts to standard form:
    0-47 = note_offs for notes that are <note+36>
    48-95 = note_ons for notes that are <note+36>
    96-<96+nClassesTimes> = Waiting times with units = <smallestTimeUnit> of a whole note


    Parameters
    ----------
    piece: list(int)
        A list representation of piece that is compatible with child class

    smallestTimeUnit: float
        smallest fraction of a whole beat that the piece is capable of capturing

    """

    @staticmethod
    def play(piece, smallestTimeUnit = 1/32, tempo = 120):
        
        timeUnitSeconds =  (smallestTimeUnit/(1/4))*(60/tempo)     #How many beats in smallest time unit
        fs = fluidsynth.Synth()
        fs.start()
        sfid = fs.sfload(sf2)
        fs.program_select(0, sfid, 0, 0)
        for msg in piece:
            
            if(msg>=Piece.TIME_UNIT_START):
                time.sleep((msg-(Piece.TIME_UNIT_START-1))*timeUnitSeconds)
            elif(msg>=Piece.N_POSSIBLE_NOTES):
                fs.noteon(0, msg-Piece.N_POSSIBLE_NOTES, 100)
            else:
                fs.noteoff(0, msg)





#Abstract class for piece
#Piece only records timing and notes
#2 purposes include storing piece and converting to standard format
class Piece:

    TIME_UNIT_START = 300
    N_POSSIBLE_NOTES = 150

    def __init__(self, piece, smallestTimeUnit):
        self.smallestTimeUnit = smallestTimeUnit
        piece = self.convertToOnOff(piece)
        self.piece = self._removeDup(piece)

    #If piece generates multiple note of the same pitch played at same time
    #This function removes them. Systematically searches until previous time unit for same note
    def _removeDup(self, piece):
        for i, note in enumerate(piece):
            if(note<Piece.TIME_UNIT_START):
                back = 1
                while(i-back>=0 and piece[i-back]<Piece.TIME_UNIT_START):
                    if(note == piece[i-back]):
                        del piece[i]
                        break
                    back+=1
        return piece


    


    #Must define convertToOnOff in child classes
    def convertToOnOff(self, piece):
        raise NotImplementedError("Must define convertToOnOff function.")

    def play(self, tempo = 120):
        """
        Plays piece after converted to OnOff form

        Parameters
        ----------
        tempo: int
            Tempo in beats per minute
        """
        Player.play(self.piece, self.smallestTimeUnit, tempo)

    def save(self, path, tempo = 120):
        """
        Saves piece as midi

        Parameters
        ----------
        path: str
            Path at which midi will be saved in. Add extension .mid
        tempo: int
            Tempo in beats per minute
        """
        timeUnitSeconds =  (self.smallestTimeUnit/(1/4))*(60/tempo)
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        for i, message in enumerate(self.piece):
            if(i>0 and self.piece[i-1]>Piece.TIME_UNIT_START-1):
                dt = int((self.piece[i-1] - (Piece.TIME_UNIT_START-1)) * timeUnitSeconds / 60 * tempo * mid.ticks_per_beat)
            else:
                dt = 0
            self._addMessage(dt, message, track)
        
        mid.save(path)

    def _addMessage(self, dt, message, track):
        if(message<Piece.N_POSSIBLE_NOTES):
            track.append(Message('note_off', note=message, velocity=127, time=dt))
        elif(message<Piece.TIME_UNIT_START):
            track.append(Message('note_on', note=message-Piece.N_POSSIBLE_NOTES, velocity=127, time=dt))



class OnOffPiece(Piece):
    ##Standard format for pieces

    #NoteOns -> 150 + <pitch>
    #NoteOffs -> 0 + <pitch>
    #Wait Time -> 299 + <number of time units>
    

    def __init__(self, piece, smallestTimeUnit):
        super().__init__(piece, smallestTimeUnit)

    

    #Already in standard form
    def convertToOnOff(self, piece):
 
        return piece




import itertools


class MultiNetPiece(Piece):

    TIME_UNIT = 300

    #Each note broken down into:
    # <pitch number> <time units player for>
    # 300 is waiting time unit

    def __init__(self, piece, smallestTimeUnit):
        super().__init__(piece, smallestTimeUnit)

    
    def convertToOnOff(self, piece):
        piece = list(itertools.chain.from_iterable(piece))
        totalTimeUnits = sum([piece[i+1] for i in range(len(piece)) if i%2 == 0 and piece[i]==Piece.TIME_UNIT_START])+100
        notesByTimeUnit = self._calcNoteOnNoteOffs(piece, totalTimeUnits)
        convertedPiece = self._collapseTimeUnits(notesByTimeUnit)
        return convertedPiece


    

    def _calcNoteOnNoteOffs(self, piece, totalTimeUnits):
        notesByTimeUnit = [[] for i in range(totalTimeUnits)]
        currentTimeUnit = 0
        for i,evt in enumerate(piece):
            if(i%2 == 0):
                if(evt==MultiNetPiece.TIME_UNIT):
                    currentTimeUnit += piece[i+1]
                else:
                    notesByTimeUnit[currentTimeUnit].append(Piece.N_POSSIBLE_NOTES+(evt))
                    notesByTimeUnit[currentTimeUnit+piece[i+1]].append(evt)    #Signals note off
        return notesByTimeUnit

    #If there are many 176's right next to each other they can be combined
    def _collapseTimeUnits(self, notesByTimeUnit):
        convertedPiece = []
        for timeUnit in notesByTimeUnit:
            if(len(timeUnit)==0 and len(convertedPiece)>0):
                convertedPiece[-1]+=1
            else:
                for note in timeUnit:
                    convertedPiece.append(note)
                convertedPiece.append(MultiNetPiece.TIME_UNIT)    #Each timeunit represents 300
        return convertedPiece


