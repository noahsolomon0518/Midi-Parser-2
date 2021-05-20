#Keras data generator classes for each type of neural network


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import keras
import numpy as np
from .pieces import MultiNetPiece
import itertools




class DataGen(keras.utils.Sequence):
    def __init__(self, encodedMidis, batchSize, lookback, gap):
        """
        Abstraction for data generators

        Parameters
        ----------
        encodedMidis: list[list]
            List of encoded pieces
        noteRange: list
            List of all the possible notes
        timeRange: list
            List of all the possible times
        batchSize: int
            Number of samples that will be evaluated before neural network adjusts weights
        lookback: int
            Number of previous notes that the network has before it makes a predictions
        gap: int
            Interval at which new samples are generated in a piece
        """
        self.encodedMidis = encodedMidis
        self.batchSize = batchSize
        self.lookback = lookback
        self.eventRange = self._calculateRange()
        self.gap = gap
        self.indices = self._getIndices()
        self._shuffleInds()


    def __getitem__(self, index):
        xIndices, yIndices = self._getXYIndexStarts(index)
        xEncodedSamples, yEncodedSamples = self._getXYEncodedSamples(xIndices,yIndices)
        return self._oneHotEncodeBatch(xEncodedSamples, yEncodedSamples)


    def _oneHotEncodeBatch(self, xEncodedSamples, yEncodedSamples):
        raise NotImplementedError("Must include _oneHotEncodeBatch() function")

    def _calculateRange(self):
        _range = []
        nFeatures = len(self.encodedMidis[0][0]) if type(self.encodedMidis[0][0]) == list else 1
        if(nFeatures>1):
            for i in range(nFeatures):
                eventsPerPiece = [list(np.array(piece)[:,i]) for piece in self.encodedMidis]
                rangeEvents= list(set(itertools.chain.from_iterable(eventsPerPiece)))
                rangeEvents.sort()
                _range.append(rangeEvents)
        else:
            rangeEvents = list(set(itertools.chain.from_iterable(self.encodedMidis)))
            rangeEvents.sort()
            _range.append(rangeEvents)
        return _range

    #Gets x and y indices starting point in form of (<piece Ind>, <sample Ind>) based on batchsize
    def _getXYIndexStarts(self, index):
        xIndices = self.indices[index*self.batchSize:(index+1)*self.batchSize]
        yIndices = [(xInd[0], xInd[1]+self.lookback) for xInd in xIndices]
        return xIndices,yIndices

    def _getXYEncodedSamples(self, xIndices, yIndices):
        xEncodedSamples = [np.array(self.encodedMidis[xStartInd[0]])[xStartInd[1]:xStartInd[1]+self.lookback] for xStartInd in np.array(xIndices)]
        yEncodedSamples = [self.encodedMidis[yInd[0]][yInd[1]] for yInd in np.array(yIndices)]
        return np.stack(xEncodedSamples), np.stack(yEncodedSamples)

    #Grabs every possible sample starting place as list of (<piece ind>, <sample ind>)
    def _getIndices(self):
        indicesByPiece = [self._getIndicesByPiece(piece) for piece in self.encodedMidis]
        indicesByPieceAndSample = [self._getIndicesByPieceAndSample(indicesByPiece[i], i) for i in range(len(indicesByPiece))]
        return list(itertools.chain.from_iterable(indicesByPieceAndSample))

    #Grabs every sample ind in a single piece
    def _getIndicesByPiece(self, piece):
        pieceLength = len(piece)
        return [i for i in range(pieceLength) if i%self.gap == 0 and (i+1)< pieceLength-self.lookback]
    
    def _getIndicesByPieceAndSample(self, indices, pieceInd):
        return [(pieceInd, i) for i in indices]

    def _shuffleInds(self):
        np.random.shuffle(self.indices)

    def on_epoch_end(self):
        self._shuffleInds()

    def __len__(self):
        distinctSamples = len(self.indices)
        return distinctSamples//self.batchSize




class DataGenOnOff(DataGen):

    def __init__(self, encodedMidis,  batchSize, lookback, gap):
        super().__init__(encodedMidis, batchSize=batchSize, lookback=lookback, gap=gap)
        self.ohe = OneHotEncoder(categories=self.eventRange, sparse=False)
        

    def _oneHotEncodeBatch(self, xEncodedSamples, yEncodedSamples):
        xSamples = np.array([self.ohe.fit_transform(sample.reshape(-1,1)) for sample in xEncodedSamples])
        ySamples = self.ohe.fit_transform(yEncodedSamples.reshape(-1,1))
        return xSamples, ySamples
        







class DataGenMultiNet(DataGen):

    def __init__(self, encodedMidis,  batchSize, lookback, gap):
        super().__init__(encodedMidis, batchSize=batchSize, lookback=lookback, gap=gap)
        self.ohe = OneHotEncoder(categories=self.eventRange, sparse=False)

        self.nClassesNotes = len(self.eventRange[0])
        self.nClassesTimes = len(self.eventRange[1])

    def _oneHotEncodeBatch(self, xEncodedSamples, yEncodedSamples):
        xSamples = np.array([self.ohe.fit_transform(sample) for sample in xEncodedSamples])
        y = self.ohe.fit_transform(yEncodedSamples)
        yNotes = y[:,:self.nClassesNotes]
        yTimes = y[:,self.nClassesNotes:]
        return xSamples, [yNotes,yTimes]





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




