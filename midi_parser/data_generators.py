#Keras data generator classes for each type of neural network


from numpy.core.fromnumeric import sort
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import keras
import numpy as np
from .pieces import MultiNetPiece
import itertools
import copy



class DataGen(keras.utils.Sequence):
    def __init__(self, encodedMidis, batchSize, lookback, gap):
        """
        Abstraction for data generators

        Parameters
        ----------
        encodedMidis: list[list]
            List of encoded pieces
        ranges: dict[list]
            Includes the ranges for notes
        batchSize: int
            Number of samples that will be evaluated before neural network adjusts weights
        lookback: int
            Number of previous notes that the network has before it makes a predictions
        gap: int
            Interval at which new samples are generated in a piece
        dimensions: list
            List of the different number of classes for each category
        """
        self.encodedMidis = encodedMidis
        self.ranges = self.getRanges()
        self.batchSize = batchSize
        self.lookback = lookback
        self.gap = gap
        self.indices = self.getIndices()
        self.shuffleInds()
        self.updateEncoders()

    def getRanges(self):
        if(type(self.encodedMidis[0][0])!=list):
            print([list(set(itertools.chain.from_iterable(self.encodedMidis)))])
            return [list(set(itertools.chain.from_iterable(self.encodedMidis)))]
        ranges = []
        for i in range(len(self.encodedMidis[0][0])):
            currentRange = sort(list(set(np.array(list(itertools.chain.from_iterable(self.encodedMidis)))[:,i])))
            ranges.append(currentRange)
        return ranges



    def validationSplit(self, test_size):
        print(np.array(self.encodedMidis))
        nTestSamples = int(test_size*len(self.encodedMidis))
        self.shuffleInds()
        testSamples, self.encodedMidis = self.encodedMidis[:nTestSamples], self.encodedMidis[nTestSamples:]
        self.indices = self.getIndices()
        validationDatagen = copy.deepcopy(self)
        validationDatagen.encodedMidis = testSamples
        validationDatagen.indices = validationDatagen.getIndices()
        return validationDatagen


    def updateEncoders(self):
        raise NotImplementedError("Must include updateEncoders()")

    def __getitem__(self, index):
        xIndices, yIndices = self._getXYIndexStarts(index)
        xEncodedSamples, yEncodedSamples = self._getXYEncodedSamples(xIndices,yIndices)
        return self._encodeBatch(xEncodedSamples, yEncodedSamples)


    def _encodeBatch(self, xEncodedSamples, yEncodedSamples):
        raise NotImplementedError("Must include _encodeBatch() function")



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
    def getIndices(self):
        indicesByPiece = [self._getIndicesByPiece(piece) for piece in self.encodedMidis]
        indicesByPieceAndSample = [self._getIndicesByPieceAndSample(indicesByPiece[i], i) for i in range(len(indicesByPiece))]
        return list(itertools.chain.from_iterable(indicesByPieceAndSample))

    #Grabs every sample ind in a single piece
    def _getIndicesByPiece(self, piece):
        pieceLength = len(piece)
        return [i for i in range(pieceLength) if i%self.gap == 0 and (i+1)< pieceLength-self.lookback]
    
    def _getIndicesByPieceAndSample(self, indices, pieceInd):
        return [(pieceInd, i) for i in indices]

    def shuffleInds(self):
        np.random.shuffle(self.indices)

    def on_epoch_end(self):
        self.shuffleInds()

    def __len__(self):
        distinctSamples = len(self.indices)
        return distinctSamples//self.batchSize





class DataGenOnOffNet(DataGen):

    def __init__(self, encodedMidis,   batchSize, lookback, gap):
        super().__init__(encodedMidis,  batchSize=batchSize, lookback=lookback, gap=gap)
    
    def updateEncoders(self):
        self.ohe = OneHotEncoder(categories=self.ranges, sparse=False)
        

    def _encodeBatch(self, xEncodedSamples, yEncodedSamples):
        xSamples = np.array([self.ohe.fit_transform(sample.reshape(-1,1)) for sample in xEncodedSamples])
        ySamples = self.ohe.fit_transform(yEncodedSamples.reshape(-1,1))
        return xSamples, ySamples
        

class DataGenEmbeddedOnOffNet(DataGen):

    def __init__(self, encodedMidis,  batchSize, lookback, gap):
        super().__init__(encodedMidis, batchSize=batchSize, lookback=lookback, gap=gap)

    def updateEncoders(self):
        self.ordEnc = OrdinalEncoder(categories=self.ranges)
        self.ohe = OneHotEncoder(categories=self.ranges, sparse=False)

    def _encodeBatch(self, xEncodedSamples, yEncodedSamples):
        x = np.array([self.ordEnc.fit_transform(sample.reshape(-1,1)) for sample in xEncodedSamples]).reshape(self.batchSize, self.lookback)
        y = self.ohe.fit_transform(yEncodedSamples.reshape(-1,1))
        return x,y




class DataGenMultiNet(DataGen):

    def __init__(self, encodedMidis, batchSize, lookback, gap):
        super().__init__(encodedMidis, batchSize=batchSize, lookback=lookback, gap=gap)

    
    def updateEncoders(self):
        self.ohe = OneHotEncoder(categories=self.ranges, sparse=False)
        self.nClassesNotes = len(self.ranges[0])
        self.nClassesTimes = len(self.ranges[1])

    def _encodeBatch(self, xEncodedSamples, yEncodedSamples):
        xSamples = np.array([self.ohe.fit_transform(sample) for sample in xEncodedSamples])
        y = self.ohe.fit_transform(yEncodedSamples)
        yNotes = y[:,:self.nClassesNotes]
        yTimes = y[:,self.nClassesNotes:]
        return xSamples, [yNotes,yTimes]





class DataGenEmbeddedMultiNet(DataGenMultiNet):
    def __init__(self, encodedMidis,  batchSize, lookback, gap):
        super().__init__(encodedMidis, batchSize=batchSize, lookback=lookback, gap=gap)
        

    def updateEncoders(self):
        self.nClassesNotes = len(self.ranges[0])
        self.nClassesTimes = len(self.ranges[1])
        self.ordEnc = OrdinalEncoder(categories=self.ranges)
        self.ohe = OneHotEncoder(categories=self.ranges, sparse=False)


    def _encodeBatch(self, xEncodedSamples, yEncodedSamples):
        x = np.array([self.ordEnc.fit_transform(sample) for sample in xEncodedSamples])
        y = self.ohe.fit_transform(yEncodedSamples)

        xNotes = x[:,:,0]
        xTimes = x[:,:,1]
        yNotes = y[:,:self.nClassesNotes]
        yTimes = y[:,self.nClassesNotes:]
        return (xNotes,xTimes), (yNotes,yTimes)




class DataGenGuideNet(DataGen):

    def __init__(self, encodedMidis, batchSize, lookback, gap):
        super().__init__(encodedMidis, batchSize=batchSize, lookback=lookback, gap=gap)
        
    
    def updateEncoders(self):
        self.nClassesNotes = len(self.ranges[0])
        self.nClassesTimes = len(self.ranges[1])
        self.ordEnc = OrdinalEncoder(categories=self.ranges)
        self.ohe = OneHotEncoder(categories=self.ranges, sparse=False)


    def _getXYEncodedSamples(self, xIndices, yIndices):
        xEncodedSamples = [np.array(self.encodedMidis[xStartInd[0]])[xStartInd[1]:xStartInd[1]+self.lookback] for xStartInd in np.array(xIndices)]
        yEncodedSamples = [np.array(self.encodedMidis[xStartInd[0]])[xStartInd[1]+1:xStartInd[1]+self.lookback+1] for xStartInd in np.array(xIndices)]
        return np.stack(xEncodedSamples), np.stack(yEncodedSamples)

    def _encodeBatch(self, xEncodedSamples, yEncodedSamples):
        x = np.array([self.ordEnc.fit_transform(sample) for sample in xEncodedSamples])
        y = np.array([self.ohe.fit_transform(sample) for sample in yEncodedSamples])

        xNotes = x[:,:,0]
        xTimes = x[:,:,1]
        yNotes = y[:,:,:self.nClassesNotes]
        yTimes = y[:,:,self.nClassesNotes:]

        return (xNotes,xTimes), (yNotes,yTimes)

