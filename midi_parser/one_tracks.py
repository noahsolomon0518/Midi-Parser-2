#One tracks convert the midi files to a form more easily encoded. OTEncoders take one tracks and are the final step 
#of decimal encoding

import math


class Note:
    def __init__(self, note, time, type, velocity, instrument):
        """
        Note data structure

        Parameters
        ----------
        note: int
            Pitch number

        time: int or float
            Some representation of time depending on the context

        type: str
            Can either be "note_on", "note_off", or "time_unit"

        velocity: int
            Hard loud the note is

        instrument: ?
            The instrument that the note is played by
        
        """
        self.note = note
        self.time = time
        self.instrument = instrument
        self.velocity = velocity
        if(velocity == 0):
            self.type = "note_off"
        else:
            self.type = type

    def copy(self):
        return Note(self.note, self.time, self.type, self.velocity, self.instrument)

#Used for OnOff decimal encoder. Also an abstraction for other one tracks. Onetracks are flattened midis
class OneTrack:

    MIDDLE_C = 60
    NOTES_PER_OCTAVE = 12
    DEFAULT_MICRO_SECS_PER_BEAT = 50000
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


    
    def __init__(self, mido, smallestTimeUnit, noteRange = [36,84], convertToC = True, scales = "both"):
        assert len(noteRange) == 2
        """
        Abstract class to extract notes from mido.
        To combine all tracks to one track (hence the name), first all note times are computed absolutely. 
        Then after that note times are converted back to relative time. 

        Parameters
        ----------
        mido: obj
            mido object that is previously instiantiated
        
        smallestTimeUnit: float
            smallest fraction of a whole note that can be captured

        noteRange: tuple
            Minimum note and maximum note respectively

        convertToC: bool
            If should convert to C. If so, does not parse midis without key signatures

        scales: str
            Can be "both", "minor", or "major". If major or minor then does not parse midis without key signatures


        """

        assert type(smallestTimeUnit) == float
        self.mido = mido
        self.smallestTimeUnit = smallestTimeUnit

        self.minNote, self.maxNote = noteRange
        self.convertToC = convertToC
        self.scales = scales
        self.key = None
        self.tpb = mido.ticks_per_beat
        self.notesAbs = []
        self.notesRel = []
        self._extractNotesAbs()

    


        #Only does full conversion if valid. Otherwise self.noteRel=[] and is not parsed
        if(self._isValid()):
            self.needKey = (self.convertToC == True or self.scales != "both")
            if(self.needKey):
                self.halfStepsAboveC = OneTrack.halfStepsAboveC[self.key.replace("m", "")]
                self.halfStepsBelowC = 14 - OneTrack.halfStepsAboveC[self.key.replace("m", "")]
            self._convertToNotesRel()
            self._applyMinMaxOctave()

    
    #Checks if midi file has key information which is needed for key change
    def _isValid(self):
        if(self.key==None and (self.convertToC == True or self.scales != "both")):
            return False
        if(self.scales=="major" and "m" in self.key):
            return False
        if(self.scales=="minor" and "m" not in self.key):
            return False
        return True



    def _extractNotesAbs(self):

        for track in self.mido.tracks:
            _time = 0
            instrument = 0
            for msg in track:
                if(msg.type == "program_change"):
                    instrument = msg.program

                if(msg.type == "key_signature"):
                    self.key = msg.key

                _time += msg.time
                
                if(msg.type in ["note_on","note_off"]):
                    self.notesAbs.append(Note(msg.note,
                                              _time,
                                              msg.type,
                                              msg.velocity, 
                                              instrument))


        self.notesAbs.sort(key=lambda x: x.time)

    #Input notesrel and changes note take in account of min max octave
    def _minMaxOctaveConvert(self, note):
        if(note.note>self.maxNote):
            octavesToShift = (((note.note-self.maxNote)//OneTrack.NOTES_PER_OCTAVE) + 1)
            note.note = int(note.note - 12 * octavesToShift)
        elif(note.note<self.minNote):
            octavesToShift = (((self.minNote - note.note)//OneTrack.NOTES_PER_OCTAVE) + 1)
            note.note = int(note.note + 12 * octavesToShift)

    #After notes converted to relative function is used to make sure every note is inside min-max note range
    def _applyMinMaxOctave(self):
        for note in self.notesRel:
            self._minMaxOctaveConvert(note)
            


    def _convertToNotesRel(self):
        notesAbs = self.notesAbs.copy()
        if(len(self.notesAbs)==0):
            return
        firstNote = notesAbs[0]
        firstNote.time = 0
        self.notesRel.append(firstNote)

        for i in range(len(notesAbs[1:])-1):
            currentNote = notesAbs[i+1]
            previousNote = notesAbs[i]
            deltaTime = currentNote.time - previousNote.time            
            currentNoteCopy = currentNote.copy()
            if(self.convertToC):
                currentNoteCopy.note = self._convertNoteToC(currentNoteCopy.note) 
            currentNoteCopy.time = self._timeConversion(deltaTime)
            self.notesRel.append(currentNoteCopy)


    def _convertNoteToC(self, note):
        newNote = note - self.halfStepsBelowC
        if((note>87 and newNote<88) or newNote<0):
            newNote = note + self.halfStepsAboveC
        return newNote


    #Can define custom time conversions in different implementations
    def _timeConversion(self, _time):
        return int(math.ceil((_time*(1/self.tpb)/4)/self.smallestTimeUnit))


class OneTrackOnOnly(OneTrack):
    """
    Calculates time each note is played for. 300 is signal for waiting time.
    """
    def __init__(self, mido, convertToC = True, scales = "both", noteRange = [36,84], smallestTimeUnit = 1/32):
        super().__init__(mido, convertToC = convertToC, scales = scales, noteRange=noteRange, smallestTimeUnit = smallestTimeUnit)
        self.notesTimed = []
        self._calculateNoteOns()

    def _calculateNoteOns(self):
        for i, note in enumerate(self.notesRel):
            if(note.time>0):
                self.notesTimed.append(Note(300, note.time, "time_unit", 0, 0 ))
            if(note.type == "note_on"):
                noteNum = note.note
                dt = 0
                for nextNote in self.notesRel[i:]:
                    if(nextNote.type == "note_off" and nextNote.note == noteNum):
                        dt+=nextNote.time 
                        break
                    dt+=nextNote.time 
                self.notesTimed.append(Note(note.note, dt, "note_on", note.velocity, note.instrument))



class OneTrackOnOnlyOrdered(OneTrackOnOnly):
    """
    Same as OneTrackOnOnly but also orders notes that are between time units and removes duplicates
    This might prevent the neural network from generating garbage because there is more consistency
    """
    def __init__(self, mido, convertToC = True, scales = "both", noteRange = [36,84], smallestTimeUnit = 1/32):
        super().__init__(mido, convertToC = convertToC, scales = scales, noteRange=noteRange, smallestTimeUnit = smallestTimeUnit)
        self._orderAll()
        self._dropDuplicates()
    
    #Used the VERY inefficient bubble sort whoops
    def _orderAll(self):
        length = len(self.notesTimed)
        isDone = False
        while not isDone:
            isDone = True
            for i, note in enumerate(self.notesTimed):
                if self._orderNote(length, i) == False:
                    isDone = False



    #Positions one note where > than previous element and < next element
    def _orderNote(self, length, i):
        note = self.notesTimed[i].note
        isDone = True
        if note<300 and i<(length-1) and note>self.notesTimed[i+1].note:
            c = 1
            while(note>self.notesTimed[i+c].note):
                self.notesTimed[i+c].note, self.notesTimed[i+c-1].note = self.notesTimed[i+c-1].note, self.notesTimed[i+c].note
                c+=1
                isDone = False
             
        elif note<300 and i>0 and note<self.notesTimed[i-1].note:
            c = 1
            while(note<self.notesTimed[i-c].note and self.notesTimed[i-c].note<300):
                self.notesTimed[i-c].note, self.notesTimed[i-c+1].note = self.notesTimed[i-c+1].note, self.notesTimed[i-c].note
                c+=1
                isDone = False
        return isDone

    def _dropDuplicates(self):
        for i, note in enumerate(self.notesTimed):
            if(note.note<300 and i>0 and i<(len(self.notesTimed)-1)):
                currentNote = self.notesTimed[i].note
                nextNote = self.notesTimed[i+1].note
                previousNote = self.notesTimed[i-1].note
                if(currentNote==nextNote or currentNote==previousNote):
                    self.notesTimed.remove(note)
            



