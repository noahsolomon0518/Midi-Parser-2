#encode onetracks to integers 

import numpy as np


class OTEncoder:
    def __init__(self, oneTracks, nClassesTimes):
        """
        Abstraction for all OTEncoders

        Paramaters
        ----------

        oneTracks: OneTrack
            List of OneTrack objects of compatible type
        """
        self.nClassesTimes = nClassesTimes
        self.encodedOTs = []
        self._encodeAll(oneTracks)
    

    def _encodeAll(self, oneTracks):
        for track in oneTracks:
            encodedOT = self._encodeOneMido(track)
            if (encodedOT != None):
                self.encodedOTs.append(encodedOT)
            

    def _encodeOneMido(self, track):
        pass





class OTEncoderOnOff(OTEncoder):
    POSSIBLE_N_NOTES = 150 #number of distinct pitchs capable of being encoded
    TIME_UNIT_START = 300 #time units are encoded as 300 - 300+<maxTimeUnit>

    def __init__(self, oneTracks, nClassesTimes):
        super().__init__(oneTracks, nClassesTimes = nClassesTimes)      #Min note extracted from onetrack


    def _encodeOneMido(self, oneTrack):

        encodedOT = []

        for note in oneTrack.notesRel:
            encodedOT.extend(self._encodeOneNote(note))
        if(len(encodedOT) != 0):
            return encodedOT

    
    def _encodeOneNote(self, note):
        waitTime = []
        waitTime = [np.min([OTEncoderOnOff.TIME_UNIT_START-1+note.time, OTEncoderOnOff.TIME_UNIT_START-1 + self.nClassesTimes])] if note.time > 0 else []
        if(note.type == "note_on"):
        
            if(note.velocity == 0):
                waitTime.append(note.note)
                return waitTime
            else:
                waitTime.append(note.note+OTEncoderOnOff.POSSIBLE_N_NOTES)
                return waitTime
        else:
            waitTime.append(note.note)
            return waitTime





class OTEncoderMultiNet(OTEncoder):

    TIME_UNIT = 300

    def __init__(self, oneTracks, nClassesTimes):
        super().__init__(oneTracks, nClassesTimes)

  
    def _encodeOneMido(self, OT):
        encodedNotes = []
        encodedTimes = []
        for note in OT.notesTimed:
    
            encodedNote, encodedTime = self._encodeOneNote(note)
            if(encodedNote!=None and encodedTime!=None):

                if(len(encodedNotes)>0 and encodedNotes[-1]==encodedNote and encodedNote == 300 and encodedTime+encodedTimes[-1]<self.nClassesTimes):
                    encodedTimes[-1]+=encodedTime
                else:
                    encodedNotes.append(encodedNote)
                    encodedTimes.append(encodedTime)
            
        if(len(encodedNotes) != 0):
            return [encodedNotes, encodedTimes]



    #Encode note to time unit or note as well as time player for
    def _encodeOneNote(self, note):
        if(note.time == 0):
            return [note.note, None]
        elif(note.type == "note_on"):
            return [note.note, np.min([note.time, self.nClassesTimes-1])]
        else:
            return [OTEncoderMultiNet.TIME_UNIT,  np.min([note.time, self.nClassesTimes-1])]