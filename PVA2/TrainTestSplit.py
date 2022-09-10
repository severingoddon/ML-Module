import numpy as np

class TrainTest:

    def __init__(self):
        pass

    @staticmethod
    def getTrainTestSplit():
        dataset = np.loadtxt("diagnosis.csv", delimiter=',')
        np.random.seed(42)
        X = dataset[:, 1:6]  # ohne erste Spalte, da diese nicht kategorisch
        Y = dataset[:, 6]  # zweitletzte Spalte mit Klasse
        allData = np.arange(X.shape[0])
        iTesting = np.random.choice(X.shape[0], int(X.shape[0] * 0.2), replace=False)
        iTraining = np.delete(allData, iTesting)
        dataRecords = len(iTraining)
        XTrain = X[iTraining, :]
        YTrain = Y[iTraining]
        return [XTrain,YTrain, dataRecords, iTesting, iTraining, X, Y]