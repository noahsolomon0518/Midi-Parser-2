#Encodes all queued midis to list of integers

from midi_parser.midi_parser import RelativeNote, DurationalNote
from mido import MidiFile


from os import walk
from os import path



class DecimalEncoder:
    def __init__(self, parsedMidis):
        """
        Abstract class for all implementations of decimal encoders

        Parameters
        ----------
        parsedMidis: list[list]
            Midis that were parsed by MidiParser
        ranges: dict[list]
            All possible values that each dimension can get
        """
        self.parsedMidis = parsedMidis
    

    def encode(self):
        return [self._encodeOne(piece) for piece in self.parsedMidis]

        
    
    #Abstract function that decimal encoders must include.
    def _encodeOne(self, piece):
        raise NotImplementedError("Add encode() function")


class DecimalEncoderWithTimes(DecimalEncoder):
    def __init__(self, parsedMidis, nClassesTimes):
        self.nClassesTimes = nClassesTimes
        super().__init__(parsedMidis)



class DecimalEncoderOnOff(DecimalEncoderWithTimes):
    def __init__(self, parsedMidis, nClassesTimes):
        """
        note_ons -> <relative note> + 150
        note_offs -> <relative note>
        time_unit -> 299 + <time>

        ranges:
            pitches
            times
        """
        try:
            assert type(parsedMidis[0][0]) == RelativeNote
        except AssertionError:
            raise AssertionError("Parsed midis must have relative notes. Use timeMeasurement = \"relative\" in MidiParser init") 
        super().__init__(parsedMidis, nClassesTimes)
        
    

    def _encodeOne(self, piece):
        oneEncoded = []
        for note in piece:
            if(note.time>0):
                oneEncoded.append(299+min([note.time, self.nClassesTimes]))     
            if(note.type == "note_on"):
                oneEncoded.append(150+note.pitch)
            else:
                oneEncoded.append(note.pitch)
        return self._order(oneEncoded)
    

    #Orders between time units for consistency reasons
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
    



class DecimalEncoderMultiNet(DecimalEncoderWithTimes):
    def __init__(self, parsedMidis, nClassesTimes):
        """
        Note ons -> [<pitch>, <duration>]
        """
        try:
            assert type(parsedMidis[0][0]) == DurationalNote
        except AssertionError:
            raise AssertionError("Parsed midis must have durational notes. Use timeMeasurement = \"durational\" in MidiParser init") 
        
        super().__init__(parsedMidis, nClassesTimes)
      


    def _encodeOne(self, piece):
        encoded = []
        for i,note in enumerate(piece):
    
            note = self._encodeEvent(note)
            if(i>0 and note[0]==300 and encoded[-1][0]==300):
                encoded[-1][1]+=note[1]
                continue

            searchInd = 0
            lastInd = len(encoded)-1
            while(lastInd-searchInd>=0 and note[0] < encoded[lastInd-searchInd][0] and encoded[lastInd-searchInd][0]!=300):
                searchInd += 1
            if(lastInd-searchInd>=0 and note[0] == encoded[lastInd-searchInd][0]):
                continue
            encoded.insert(lastInd+1 - searchInd, note)
                

        return encoded

    
    def _encodeEvent(self, evt):
        if(evt.type == "time_unit"):
            return [300, evt.time]
        return [evt.pitch, evt.time]





class DecimalEncoderMiniBachStyle(DecimalEncoder):
    """
    Seperated out into each time unit and the notes that are played
    """


    def __init__(self, parsedMidis):
        try:
            assert type(parsedMidis[0][0][0]) == RelativeNote
        except AssertionError:
            raise AssertionError("Parsed midis must have durational notes. Use timeMeasurement = \"durational\" in MidiParser init") 
        super().__init__(parsedMidis)
    
    def _encodeOne(self, piece):
        encoded = []
        for timeUnit in piece:
            timeUnit.sort(key = lambda x: x.pitch)
            if(len(timeUnit)==0):
                encoded.append(["none"])
            else:
                encoded.append([self._encodeEvt(note) for note in timeUnit])
        return encoded

    def _encodeEvt(self, evt):
        return evt.type+"_"+str(evt.pitch)


