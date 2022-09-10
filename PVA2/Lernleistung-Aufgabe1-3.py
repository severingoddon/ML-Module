from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


clf = GaussianNB()
neigh = KNeighborsClassifier(n_neighbors=3)

iris = datasets.load_iris()
digits = datasets.load_digits()

X = digits.data # currently using digits, change to iris when using iris
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)
neigh.fit(X_train, y_train)

print(neigh.predict(X_test))
print(neigh.score(X_test, y_test))
print(clf.predict(X_test))
print(clf.score(X_test, y_test))





