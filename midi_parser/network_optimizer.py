
from keras.callbacks import EarlyStopping
from itertools import product


#These classes are used to fine tune the networks as well as log statistics and pieces




class GridSearch:

    def __init__(self, buildModelFn,  paramGrid):
        self.buildModelFn = buildModelFn
        self.paramGrid = paramGrid


        self.parameters =  list(paramGrid.keys())
        self.values = list(paramGrid.values())
        self.parameterCombinations = list(product(*self.values))
        self.organizedParameterCombos = []

        for paramSet in self.parameterCombinations:
            self.organizedParameterCombos.append(dict([(self.parameters[i], paramSet[i]) for i in range(len(self.parameters))]))




    def fit(self, datagenTrain, datagenValidation, path, epochs = 30):
        f = open(path, "w")
        f.write(",".join(self.parameters+["score"]))
        f.write("\n")

        for i,parameterCombination in enumerate(self.organizedParameterCombos):
            print("Grid Search: "+ str(i))
            print(" Parameters: "+ str(parameterCombination))
            model = self.buildModelFn(**parameterCombination)
            model.fit(datagenTrain, validation_data = datagenValidation, epochs = epochs, callbacks = [EarlyStopping(patience=3)], verbose = 0)
            score = model.evaluate(datagenValidation)[0]
            parameterValues = [str(value) for value in list(parameterCombination.values())]
            f.write(",".join(parameterValues+[str(score)])+"\n")
            print("   Val Loss: "+ str(score) +"\n")
        f.close()       

            
                





    

        

        