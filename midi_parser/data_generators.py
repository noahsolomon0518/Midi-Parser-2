#Keras data generator classes for each type of neural network


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import keras
import numpy as np
from .pieces import MultiNetPiece
import itertools


#Takes one hot encoded vectors and converts them to their respective integers. Good for testing
def encodeFromOneHot(generated):
    piece = []
    for note in generated:
        piece.append(np.argmax(note))
    return piece



class DataGenOnOff(keras.utils.Sequence):

    def __init__(self, encodedMidis,  batchSize=32, lookback=50, gap = 5):
        """
        Data generator for on off encoder

        Parameters
        ----------
        decimalEncoder: DecimalEncoderOnOff
            Decimal encoder that is used to extract midis and encode them to integers
        
        batchSize: int
            Size of each batch in an epoch

        lookback: int
            How many steps before current step training data has access to

        gap: int
            At which interval samples will be picked out of piece
        """
        self.encodedMidis = encodedMidis
        _range = self._calculateRange()
        print(_range)
        self.ohe = OneHotEncoder(categories = [_range], sparse = False)
        self.gap = gap
        self.lookback = lookback
        self.batchSize = batchSize
        self.indices = np.array([(pieceInd,noteInd) for pieceInd in range(len(encodedMidis)) for noteInd in range(len(self.encodedMidis[pieceInd]))  if noteInd%gap == 0 and (noteInd+1)< len(self.encodedMidis[pieceInd])-self.lookback])
        self._shuffleInds()



    #Returns all possible values that OneHotEncoder might take in    
    def _calculateRange(self):
        return list(set(itertools.chain.from_iterable([piece for piece in self.encodedMidis])))

    #Calculates how many batches to cycle through all data
    def __len__(self):
        distinctSamples = len(self.indices)
        return distinctSamples//self.batchSize


    #Annoyingly complicated way to get samples
    def __getitem__(self, index):
        xIndices = self.indices[index*self.batchSize:(index+1)*self.batchSize]
        yIndices = np.array(list(map(lambda x: (x[0], x[1]+self.lookback),xIndices)))
        xEncoded = np.array(list(map(lambda x: self.encodedMidis[x[0]][x[1]:x[1]+self.lookback], xIndices)))
        yEncoded = np.array(list(map(lambda y: self.encodedMidis[y[0]][y[1]], yIndices)))
        X, y = self.__data_generation(xEncoded, yEncoded)

        return X, y



    def __data_generation(self, xEncoded, yEncoded):
        # one hot encode sequences

        x = np.array([self.ohe.fit_transform(sample.reshape(-1,1)) for sample in xEncoded])
        y = self.ohe.fit_transform(yEncoded.reshape(-1,1))
        return (x, y)




    def _shuffleInds(self):
        np.random.shuffle(self.indices)


    def on_epoch_end(self):
        self._shuffleInds()

class DataGenMultiNet(keras.utils.Sequence):

    def __init__(self, encodedMidis,  batchSize=32, lookback=50, gap = 5):

        """
        Data generator for multi net encoder

        Parameters
        ----------
        decimalEncoder: DecimalEncoderMultiNet
            Decimal encoder that is used to extract midis and encode them to integers
        
        batchSize: int
            Size of each batch in an epoch

        lookback: int
            How many steps before current step training data has access to

        gap: int
            At which interval samples will be picked out of piece
        """
        self.encodedMidis = encodedMidis
        self.gap = gap
        self.lookback = lookback
        self.batchSize = batchSize
        _range = self._calculateRange()
        print(_range)
        self.nClassesNotes = len(_range[0])
        self.indices = self._getIndices()
        self.ohe = OneHotEncoder(categories = _range, sparse = False)
        self.indices = np.array([(pieceInd,noteInd) for pieceInd in range(len(self.encodedMidis)) for noteInd in range(len(self.encodedMidis[pieceInd]))  if noteInd%gap == 0 and (noteInd+1+self.lookback)< len(self.encodedMidis[pieceInd])])
        self._shuffleInds()


    def _calculateRange(self):

        pieceNotes = [list(np.array(piece)[:,0]) for piece in self.encodedMidis]
        pieceTimes = [list(np.array(piece)[:,1]) for piece in self.encodedMidis]

        rangeNotes = list(set(itertools.chain.from_iterable(pieceNotes)))
        rangeNotes.sort()
        rangeTimes = list(set(itertools.chain.from_iterable(pieceTimes)))
        rangeTimes.sort()
        return [rangeNotes, rangeTimes]

    #Gets every possible sample. List of piece ind and starting ind in piece
    def _getIndices(self):
        return np.array([(pieceInd,noteInd) for pieceInd in range(len(self.encodedMidis)) for noteInd in range(len(self.encodedMidis[pieceInd]))  if noteInd%self.gap == 0 and (noteInd+1+self.lookback)< len(self.encodedMidis[pieceInd][0])])



    #Calculates how many batches to cycle through all data
    def __len__(self):
        distinctSamples = len(self.indices)
        return distinctSamples//self.batchSize


    #Annoyingly complicated way to get samples
    def __getitem__(self, index):
        xIndices = self.indices[index*self.batchSize:(index+1)*self.batchSize]
        yIndices = np.array(list(map(lambda x: (x[0], x[1]+self.lookback),xIndices)))

        xEncoded = np.array(list(map(lambda x: self._mapX(x), xIndices)))
        yEncoded = np.array(list(map(lambda y: self._mapY(y), yIndices)))
        X, yNotes, yTimes = self.__data_generation(xEncoded, yEncoded)
   
        return X, [yNotes, yTimes]

    #Based on x starting indices returns sequence that will be one hot encoded
    #Returns list of [note, time]
    def _mapX(self, startInd):
        noteTimes = self.encodedMidis[startInd[0]][startInd[1]:startInd[1]+self.lookback]
        return noteTimes

    def _mapY(self, yInd):
        return self.encodedMidis[yInd[0]][yInd[1]]



    def __data_generation(self, xEncoded, yEncoded):
        x = np.array([self.ohe.fit_transform(sample) for sample in xEncoded])
        y = self.ohe.fit_transform(yEncoded)
        yNotes = y[:,:self.nClassesNotes]
        yTimes = y[:,self.nClassesNotes:]
        return (x, yNotes, yTimes)




    def _shuffleInds(self):
        np.random.shuffle(self.indices)


    def on_epoch_end(self):
        self._shuffleInds()


class DataGenEmbeddedMultiNet(DataGenMultiNet):
    def __init__(self, decimalEncoder,  batchSize=32, lookback=50, gap = 5):
        _rangeNotes = np.arange(decimalEncoder.minNote, decimalEncoder.maxNote+1)
        _rangeNotes = np.append(_rangeNotes,300)
        _rangeTimes = np.arange(0,decimalEncoder.nClassesTimes)
        self.noteEnc = LabelEncoder().fit(_rangeNotes)
        self.timeEnc = LabelEncoder().fit(_rangeTimes)
        super().__init__(decimalEncoder=decimalEncoder, batchSize=batchSize,lookback=lookback, gap=gap)


    #Annoyingly complicated way to get samples
    def __getitem__(self, index):
        xIndices = self.indices[index*self.batchSize:(index+1)*self.batchSize]
        yIndices = np.array(list(map(lambda x: (x[0], x[1]+self.lookback),xIndices)))
        

        xEncoded = np.array(list(map(lambda x: self._mapX(x), xIndices)))
        yEncoded = np.array(list(map(lambda y: self._mapY(y), yIndices)))
        xNotes, xTimes, yNotes, yTimes = self.__data_generation(xEncoded, yEncoded)
   
        return [xNotes,xTimes], [yNotes, yTimes]


    def __data_generation(self, xEncoded, yEncoded):
        #Encodes notes and times to integers so can be seperately inputted into an embedding
        y = self.ohe.fit_transform(yEncoded)
        xNotes = np.array([self.noteEnc.transform(sample[:,0]) for sample in xEncoded])
        xTimes = np.array([self.timeEnc.transform(sample[:,1]) for sample in xEncoded])
        yNotes = y[:,:self.nClassesNotes]
        yTimes = y[:,self.nClassesNotes:]
        return (xNotes, xTimes, yNotes, yTimes)




