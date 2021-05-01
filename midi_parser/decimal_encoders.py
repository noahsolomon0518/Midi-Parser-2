#Encodes all queued midis to list of integers

from mido import MidiFile
from .one_tracks import *
from .ot_encoders import *
from os import walk
from os import path
import math
import warnings




# Takes list of paths (or just one), and parses into mido object
def parseToMidos(paths):
    midos = []

    if(type(paths) != list):
        paths = [paths]

    for _path in paths:
        midos.append(MidiFile(_path, type=0))

    return midos


# Recursively creates list of midi files in directory
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



class DecimalEncoder:
    # All encoders go through 3 main steps: 
    #    1. MidiToDecimal._pathsToMido -> converts all queued up paths to mido objects
    #                                     completely encapsulated
    #
    #    2. MidiToDecimal._midosToOT -> convert all midos into OneTrack objects 
    #                                   !must add _convertMidoToOneTrack(mido) to implementation
    #
    #    3. MidiToDecimal._OTEncode -> encodes all OneTrack objects to decimal encoded list
    #                                  !must add _OTEncodeOne(oneTrack) to implementation
   
    """
    Abstract class for all implementations of decimal encoders

    Parameters
    ----------
    folder: str
        Path at which midis of interest are
    
    
    debug: bool
        Saves all steps of data transformation

    r: bool
        If to recursively extract midis from folder
    
    convertToC: bool
        If want to convert all midis to same key of C. Will not extract midis without key sigs
    
    scales: str in ["major", "minor", "both"]
        If want to only extract major, minor, or both types of keys
    """
    def __init__(self, folder, smallestTimeUnit, nClassesTimes, noteRange = (36,84), debug=False, r=True, convertToC = True,  scales = "both"):
        self.convertToC = convertToC
        self.nClassesTimes = nClassesTimes
        self.smallestTimeUnit = smallestTimeUnit
        self.minNote = noteRange[0]
        self.maxNote = noteRange[1]
        self.scales = scales
        self.midos = []
        self.oneTracks = []
        self.encoded = []
        self.paths = []
        self.addFolders(folder, r=True)
        self.debug = debug


    def addFolders(self, folder, r=True):
        self.paths.extend(findMidis(folder, r))
    """
    Queues up more folders to parse

    Parameters
    ----------
    folder: str
        Folder that will be queued
    
    r: bool
        Whether folders should be recursively extracted
    """

    #call when all midi folders added
    def encode(self):
        self._dbg("---Decimal Encoding Started---")
        self._dbg("Converting queued paths to mido")
        midos = self._pathsToMidos(self.paths)
        self._dbg("Converting midos to OneTracks")
        oneTracks = self._midosToOT(midos)
        self._dbg("OTEncoding OneTracks")
        encoded = self._OTEncode(oneTracks)
        self._dbg("Encoded "+ str(len(encoded))+ " tracks")
        if(len(encoded)==0):
            warnings.warn("No valid midis")
        return encoded
        

    def _pathsToMidos(self, paths):
        if(not self.debug):
            return parseToMidos(paths)
        self.midos = parseToMidos(paths)
        return self.midos

    def _midosToOT(self, midos):
        oneTracks = []
        for mido in midos:
            ot = self._initOneTrack(mido)
            assert isinstance(ot, OneTrack)
            if(ot.notesRel != [] and ot != None):           #Returns None if scales specified and midi does not have key
                oneTracks.append(ot)
        if(not self.debug):
            return oneTracks
        self.oneTracks = oneTracks
        return oneTracks

    def _OTEncode(self, oneTracks):
        oTEncoder = self._initOTEncoder(oneTracks)
        assert isinstance(oTEncoder, OTEncoder)
        OTEncoded = oTEncoder.encodedOTs
        if(not self.debug):
            return OTEncoded
        self.encoded = OTEncoded
        return OTEncoded

    #Should return the OTEncoder
    def _initOTEncoder(self, OTs):
        raise NotImplementedError("Must define OTEncoder that will be used")
    
    def _initOneTrack(self, mido):
        raise NotImplementedError("Must define OneTrack that will be used")
    

    def _dbg(self, msg):
        if(self.debug):
            print(msg)






class DecimalEncoderOnOff(DecimalEncoder):
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
        return OneTrack(
            mido, 
            convertToC = self.convertToC, 
            noteRange = (self.minNote, self.maxNote),
            scales = self.scales, 
            smallestTimeUnit = self.smallestTimeUnit)

    def _initOTEncoder(self, oneTracks):
        return OTEncoderOnOff(oneTracks, nClassesTimes = self.nClassesTimes)




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












