#Generates music based on trained model and dategen object

import fluidsynth
import time
import os
import numpy as np
from mido import MidiFile, MidiTrack, Message
from .pieces import *
sf2 = os.path.abspath("C:/Users/noahs/Local Python Libraries/soundfonts/piano.sf2")
TIME_UNIT_START = 300
N_POSSIBLE_NOTES = 150

#Mixes up probabilities of multinomial distribution (prediction of music neural network)
def sample(preds, temperature = 1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    expPreds = np.exp(preds)
    preds = expPreds/np.sum(expPreds)
    probs = np.random.multinomial(1,preds,1)
    return np.argmax(probs)


def sampleTop(preds, n, temperature = 1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    expPreds = np.exp(preds)
    preds = expPreds/np.sum(expPreds)
    indsSorted = np.argsort(preds)
    topProbInds = indsSorted[0,-n:]
    topProbs = preds[:,topProbInds]
    sumProbs = np.sum(topProbs[0])
    normalizedProbs = [[prob/sumProbs for prob in topProbs[0]]]


    sampledProb = np.random.multinomial(1,normalizedProbs[0],1)
    sampledProbInd = topProbInds[np.argmax(sampledProb)]
    return sampledProbInd

    










class Generator:
    """
    Abstract class for the music generators

    Parameters
    ----------
    model: Keras NN
        A trained keras neural network of compatible type

    datagen: Numpy Array
        The one hot encoded training data that was used train the NN

    smallestTimeUnit: float
        The fraction of a 4/4 measure to consider the smallest time unit 
    
    nOctaves: even int
        Number of octaves that the model was trained with. Can only be 2,4,6 or 8
    
    """
    
    def __init__(self, model, datagen, smallestTimeUnit):

        self.smallestTimeUnit = smallestTimeUnit
        self.model = model
        self.datagen = datagen
        self.lookback = datagen.lookback
        self.generated = []

    def generate(self, temp, nNotes, sampleTopProbs = 0):
        assert type(nNotes) == int
        assert type(temp) in [float, int]
        """
        Generates new music using model of compatible type

        Parameters
        ----------
        temp: float
            Float larger than 0. A temp>1 results in more random generations and temp<1 results in less random
        
        nNotes: int
            Number of notes the will be generated
        """

        generated = self._generate(temp, nNotes, sampleTopProbs=sampleTopProbs)
        self.generated.append(generated)
        return generated

    def _generate(self, temp, nNotes):
        raise NotImplementedError("_generate function must be implemented")
    

    def _getPriorNotes(self, generated):
        return np.expand_dims(generated[-self.lookback:], axis = 0)
        
    
        
        

class GeneratorOnOff(Generator):
    def __init__(self, model, datagen, smallestTimeUnit):
        super().__init__(model,datagen, smallestTimeUnit)
    
    def _generate(self, temp, nNotes = 500, sampleTopProbs = 0):
        piece = []
        choiceInd = np.random.randint(0,self.datagen.batchSize)
        generated = self.datagen.__getitem__(0)[0][choiceInd]
        for i in range(nNotes):
            priorNotes = self._getPriorNotes(generated)
            preds = self.model.predict(priorNotes)
            if sampleTop:
                argMax = sampleTop(preds, n = sampleTopProbs, temperature= temp)     #I foolishly designed the function thinking it was 2d array
            else:
                argMax = sample(preds[0], temp)
            piece.append(self.datagen.ohe.categories_[0][argMax])
            generated = np.concatenate([generated,preds])
        piece = self._convertToPieceObj(piece)
        return piece
    
    def _convertToPieceObj(self, piece):
        return OnOffPiece(piece, self.smallestTimeUnit)
    








class GeneratorMultiNet(Generator):
    def __init__(self, model, datagen, smallestTimeUnit):
        super().__init__(model,datagen, smallestTimeUnit)

    def _generate(self, temp, nNotes, sampleTopProbs):
        piece = []
        choiceInd = np.random.randint(0,self.datagen.batchSize)
        generated = self.datagen.__getitem__(0)[0][choiceInd]

        for i in range(nNotes):
            priorNotes = self._getPriorNotes(generated)
            predNotes, predTimes = self.model.predict(priorNotes)
            if sampleTop:
                argmaxNotes = sampleTop(predNotes, n = sampleTopProbs, temperature= temp)     #I foolishly designed the function thinking it was 2d array
                argmaxTimes = sampleTop(predTimes, n = sampleTopProbs, temperature= temp)     #I foolishly designed the function thinking it was 2d array

            else:
                argmaxNotes = sample(predNotes[0], temp)
                argmaxTimes = sample(predTimes[0], temp)
            note = self.datagen.ohe.categories_[0][argmaxNotes]
            time = self.datagen.ohe.categories_[1][argmaxTimes]
            piece.append([note,time])
            preds = np.concatenate([predNotes, predTimes], axis = 1)
            generated = np.concatenate([generated,preds])
        piece = self._convertToPieceObj(piece)
        return piece
    
    def _convertToPieceObj(self, piece):
        return MultiNetPiece(piece, self.smallestTimeUnit)




class GeneratorEmbeddedMultiNet(Generator):
    def __init__(self, model, datagen, smallestTimeUnit):
        super().__init__(model,datagen, smallestTimeUnit)


    def _generate(self, temp, nNotes, sampleTopProbs):
        piece = []
        choiceInd = np.random.randint(0,self.datagen.batchSize)
        print(self.datagen[0][0][0][choiceInd])
        notes = self.datagen[0][0][0][choiceInd]
        times = self.datagen[0][0][1][choiceInd]
        generated = [list(notes), list(times)]
        for i in range(nNotes):
            priorNotes = self._getPriorNotes(generated[0])
            priorTimes = self._getPriorNotes(generated[1])  
            predNotes, predTimes = self.model.predict([priorNotes, priorTimes])
            if sampleTop:
                argmaxNotes = sampleTop(predNotes, n = sampleTopProbs, temperature= temp)    
                argmaxTimes = sampleTop(predTimes, n = sampleTopProbs, temperature= temp)    

            else:
                argmaxNotes = sample(predNotes[0], temp)
                argmaxTimes = sample(predTimes[0], temp)
            note = self.datagen.ohe.categories_[0][argmaxNotes]
            time = self.datagen.ohe.categories_[1][argmaxTimes]
            piece.append((note, time))
            encodedNoteTime = self.datagen.ordEnc.fit_transform([(note,time)])[0]
            generated[0].append(encodedNoteTime[0])
            generated[1].append(encodedNoteTime[1])
        piece = self._convertToPieceObj(piece)
        return piece

    def _convertToPieceObj(self, piece):
        return MultiNetPiece(piece, self.smallestTimeUnit)


            
    def _generate2(self, temp, nNotes, sampleTopProbs): 
        piece = []
        choiceInd = np.random.randint(0,self.datagen.batchSize)
        notes = self.datagen[0][0][choiceInd]
        times = self.datagen[0][1][choiceInd]
        generated = np.stack([self.datagen.ordEnc.inverse_transform([(notes[i], times[i])]) for i in range(len(notes))]).reshape(-1,2)
        for i in range(nNotes):
            priorNotes = self._getPriorNotes(generated)
            priorTimes = self._getPriorNotes(generated)
            predNotes, predTimes = self.model.predict([priorNotes, priorTimes])
            if sampleTop:
                argmaxNotes = sampleTop(predNotes, n = sampleTopProbs, temperature= temp)    
                argmaxTimes = sampleTop(predTimes, n = sampleTopProbs, temperature= temp)    

            else:
                argmaxNotes = sample(predNotes[0], temp)
                argmaxTimes = sample(predTimes[0], temp)
            
            note = self.datagen.ohe.categories_[0][argmaxNotes]
            time = self.datagen.ohe.categories_[1][argmaxTimes]
            generated = np.concatenate([generated,[[argmaxNotes], [argmaxTimes]]], axis = 1)
        print(piece)
        piece = self._convertToPieceObj(piece)
        return piece
    
    def _convertToPieceObj(self, piece):
        return MultiNetPiece(piece, self.smallestTimeUnit)

    

   












