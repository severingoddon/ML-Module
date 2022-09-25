import numpy as np
import TrainTestSplit


class NaivBayesKategorisch:

    # get all necessary data from TrainTest class
    def __init__(self):
        data = TrainTestSplit.TrainTest.getTrainTestSplit()
        self.XTrain = data[0]
        print("XTrain: " , self.XTrain)
        print("-----------------------------------------------")
        self.YTrain = data[1]
        print("YTrain: " , self.YTrain)
        print("-----------------------------------------------")
        self.dataRecords = data[2]
        print("dataRecords: " , self.dataRecords)
        print("-----------------------------------------------")
        self.iTesting = data[3]
        print("iTesting: " , self.iTesting)
        print("-----------------------------------------------")
        self.iTraining = data[4]
        print("iTraining: " , self.iTraining)
        print("-----------------------------------------------")
        self.X = data[5]
        print("X: " , self.X)
        print("-----------------------------------------------")
        self.Y = data[6]
        print("Y: " , self.Y)
        print("-----------------------------------------------")
        self.fit()

    # train the model
    def fit(self):
        self.PXI = np.zeros((2, self.XTrain.shape[1], 2))
        for k in range(self.XTrain.shape[1]):
            self.PXI[1, k, 1] = np.sum(self.XTrain[:, k] * self.YTrain)
            self.PXI[1, k, 0] = np.sum(self.XTrain[:, k] * (1 - self.YTrain))
            self.PXI[0, k, 1] = np.sum((1 - self.XTrain[:, k]) * self.YTrain)
            self.PXI[0, k, 0] = np.sum((1 - self.XTrain[:, k]) * (1 - self.YTrain))

    # predict a class for a given element
    def predict(self,x):
        PI = np.zeros(2)
        PI[1] = np.sum(self.YTrain)
        PI[0] = self.dataRecords - PI[1]
        P = np.zeros_like(PI)
        allOfThem = np.arange(self.XTrain.shape[1])
        for i in range(len(PI)):
            P[i] = np.prod(self.PXI[i, allOfThem, x]) * PI[i]
        chosenClass = np.argmax(P)
        return chosenClass

    # score my model
    def score(self):
        XTest = self.X[self.iTesting, :]
        YTest = self.Y[self.iTesting]
        correct = np.zeros(2)
        incorrect = np.zeros(2)
        for i in range(XTest.shape[0]):
            klasse = self.predict(XTest[i, :].astype(int))
            if klasse == YTest[i]:
                correct[klasse] = correct[klasse] + 1
            else:
                incorrect[klasse] = incorrect[klasse] + 1
        print(
            f"Of {XTest.shape[0]} Testcases, {int(np.sum(correct))} have been classified right and {int(np.sum(incorrect))} have been classified wrong.")


test = NaivBayesKategorisch()
test.score()
