#encode onetracks to integers 

import numpy as np





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