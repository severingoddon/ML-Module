import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("VereinigteBilddaten.csv", header=None)
X = dataset.iloc[:, 1:101]  # without first and last column (name and label not used)
Y = dataset.iloc[:, 101]  # last column

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(1000,100), max_iter=1000, alpha=1e-4, solver='adam', verbose=10,
                    tol=1e-4, random_state=1, learning_rate_init=.1,activation='relu')


mlp.fit(X_train, y_train)

print("Test set score: {0}".format(round(mlp.score(X_test, y_test)*100)),'%') # score currently around 93%


# just for testing and fun: new image that is not in the training set and not in the testing set. (Image is a "+" and is correctly recognized).
img = [[0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0]]
print(mlp.predict(img))

