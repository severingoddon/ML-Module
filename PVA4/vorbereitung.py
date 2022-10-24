import numpy as np

np.random.seed(100)
fFloat = open("Autoklassifizierung.csv", "r")
dataset = np.loadtxt(fFloat, delimiter=",")
fFloat.close()

y = dataset[:, 0]  # from all arrays in 2d array, take the first (0) element. ":" stands for all elements
x = np.ones((len(y), 3))  # generates a 2d array filled up with ones of len(y) rows and 3 columns
x[:, 0:2] = dataset[:, 1:3]  # for all arrays (:) in x, change values at index 0 to 2 to data from dataset arrays at index 1 to 3. Result for one array in x will be eg. [26490, 77, 1]

xMin = x[:, 0:2].min(axis=0)  # find the minimum values in all arrays of x at index 0 to 2
xMax = x[:, 0:2].max(axis=0)  # find the maximum values in all arrays of x at index 0 to 2
x[:, 0:2] = (x[:, 0:2] - xMin) / (xMax - xMin)  # norming values to minimize occurence of distortion (Verzerrung). After that, values are between 0 and 1
t = 0
tmax = 100000
eta = 0.25
Dw = np.zeros(3)  # --> [0. 0. 0.]
w = np.random.rand(3) - 0.5 # randomly initialize the weights
convergenz = 1

# activation function
def myHeaviside(x):
    y = np.ones_like(x,dtype=float) # returns an array of same shape like x
    y[x <= 0] = 0
    return y

# do the training
while (convergenz > 0) and (t<tmax):
    t = t +1
    WaehleBeispiel = np.random.randint(len(y))
    xB = x[WaehleBeispiel,:].T # choose a random car in the dataset (.T because we need to do matrix multiplication and that won't work without transponieren )
    yB = y[WaehleBeispiel] # take one random value (WaehleBeispiel) of y

    # w@xB --> w1,1 · x1 + w1,2 · x2 + w1,3 · 1
    error = yB - myHeaviside(w@xB) # calculate the error.
    for j in range(len(xB)):
        Dw[j]= eta*error*xB[j]
        w[j] = w[j] + Dw[j] # adjust the weights
    convergenz = np.linalg.norm(y-myHeaviside(w@x.T))

def predict(x,w,xMin,xMax):
    xC = np.ones( (x.shape[0],3) )
    xC[:,0:2] = x
    xC[:, 0:2] = (xC[:, 0:2] - xMin) / (xMax - xMin)
    y = w @ xC.T
    y[y > 0] = 1
    y[y <= 0] = 0
    return (y)

xTest = np.array([[12490, 48], [31590, 169],[24740, 97], [30800, 156]])
yPredict = predict(xTest,w,xMin,xMax)
print(yPredict)
