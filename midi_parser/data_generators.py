#Keras data generator classes for each type of neural network


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .decimal_encoders import DecimalEncoderMultiNet2, DecimalEncoderMultiNet, DecimalEncoderOnOff
import keras
import numpy as np
from .pieces import MultiNetPiece


#Takes one hot encoded vectors and converts them to their respective integers. Good for testing
def encodeFromOneHot(generated):
    piece = []
    for note in generated:
        piece.append(np.argmax(note))
    return piece



class DataGenOnOff(keras.utils.Sequence):

    def __init__(self, decimalEncoder,  batchSize=32, lookback=50, gap = 5):
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
        _range = self._calculateRange(decimalEncoder.minNote, decimalEncoder.maxNote, decimalEncoder.nClassesTimes)
        self.ohe = OneHotEncoder(categories = [_range], sparse = False)
        self.decimalEncoder = decimalEncoder
        self.encoded = decimalEncoder.encode()
        self.gap = gap
        self.lookback = lookback
        self.batchSize = batchSize
        self.nClassesTimes = decimalEncoder.nClassesTimes
        self.indices = np.array([(pieceInd,noteInd) for pieceInd in range(len(self.encoded)) for noteInd in range(len(self.encoded[pieceInd]))  if noteInd%gap == 0 and (noteInd+1)< len(self.encoded[pieceInd])-self.lookback])
        self._shuffleInds()



    #Returns all possible values that OneHotEncoder might take in    
    def _calculateRange(self, minNote, maxNote, nClassesTimes):
        possibleNoteOffs = np.arange(minNote, maxNote+1)
        possibleNoteOns = np.arange(minNote+150, maxNote+151)
        possibleTimes = np.arange(300, 300+nClassesTimes)
        return np.concatenate([possibleNoteOffs, possibleNoteOns,  possibleTimes], axis = 0)

    #Calculates how many batches to cycle through all data
    def __len__(self):
        distinctSamples = len(self.indices)
        return distinctSamples//self.batchSize


    #Annoyingly complicated way to get samples
    def __getitem__(self, index):
        xIndices = self.indices[index*self.batchSize:(index+1)*self.batchSize]
        yIndices = np.array(list(map(lambda x: (x[0], x[1]+self.lookback),xIndices)))
        xEncoded = np.array(list(map(lambda x: self.encoded[x[0]][x[1]:x[1]+self.lookback], xIndices)))
        yEncoded = np.array(list(map(lambda y: self.encoded[y[0]][y[1]], yIndices)))
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

    def __init__(self, decimalEncoder,  batchSize=32, lookback=50, gap = 5):

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
        _rangeNotes = np.arange(decimalEncoder.minNote, decimalEncoder.maxNote+1)
        _rangeNotes = np.append(_rangeNotes,300)
        _rangeTimes = np.arange(0,decimalEncoder.nClassesTimes)
        self.ohe = OneHotEncoder(categories = [_rangeNotes, _rangeTimes], sparse = False)
        self.decimalEncoder = decimalEncoder
        self.encoded = decimalEncoder.encode()
        self.gap = gap
        self.lookback = lookback
        self.batchSize = batchSize
        self.nClassesNotes = len(_rangeNotes)
        self.nClassesTimes = decimalEncoder.nClassesTimes
        self.indices = np.array([(pieceInd,noteInd) for pieceInd in range(len(self.encoded)) for noteInd in range(len(self.encoded[pieceInd][0]))  if noteInd%gap == 0 and (noteInd+1+self.lookback)< len(self.encoded[pieceInd][0])])
        self._shuffleInds()




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
        notes = self.encoded[startInd[0]][0][startInd[1]:startInd[1]+self.lookback]
        times = self.encoded[startInd[0]][1][startInd[1]:startInd[1]+self.lookback]
        return list(zip(notes,times))

    def _mapY(self, yInd):
        note = self.encoded[yInd[0]][0][yInd[1]]
        time = self.encoded[yInd[0]][1][yInd[1]]
        return (note, time)



    def __data_generation(self, xEncoded, yEncoded):
        # one hot encode sequences
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




