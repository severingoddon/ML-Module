import numpy as np
import TrainTestSplit


class NaivBayesKategorisch:

    # get all necessary data from TrainTest class
    def __init__(self):
        data = TrainTestSplit.TrainTest.getTrainTestSplit()
        self.XTrain = data[0]
        self.YTrain = data[1]
        self.dataRecords = data[2]
        self.iTesting = data[3]
        self.iTraining = data[4]
        self.X = data[5]
        self.Y = data[6]
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
        allofthem = np.arange(self.XTrain.shape[1])
        for i in range(len(PI)):
            P[i] = np.prod(self.PXI[i, allofthem, x]) * PI[i]
        choosenClass = np.argmax(P)
        return choosenClass

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
            f"Von {XTest.shape[0]} Testfällen wurden {int(np.sum(correct))} richtig und {int(np.sum(incorrect))} falsch klassifiziert")


test = NaivBayesKategorisch()
test.score()
