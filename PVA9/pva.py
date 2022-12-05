import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

autos = pd.read_csv("Autoklassifizierung.csv", header=None)
autos.head()
scaler = MinMaxScaler()
autos_x = scaler.fit_transform(autos[[1, 2]])
autos_y = autos[0].values
svmLin = SVC(kernel='linear', C=100)
svmLin.fit(autos_x, autos_y)

def coefs(svm_):
    """
    Bestimmt die Koeffizienten der Ternngeraden und der Margingeraden
    für lineare SVM bei Datensätzen mit zwei Features.
    """
    coefs = svm_.coef_[0]
    a = -coefs[0] / coefs[1]
    b = - svm_.intercept_ / coefs[1]
    margin = 1 / np.linalg.norm(coefs)
    margin_y_offset = margin * (a ** 2 + 1) ** 0.5
    return a, b, b - margin_y_offset, b + margin_y_offset


def mesh(svm_, xmin=0, xmax=1, ymin=0, ymax=1, colors='cygmbr'):
    """
    Erzeugt zwei Arrays mit x- und y-Koordinaten, die relativ dicht
    in einenm Rechteck liegen.
    Ferner wird wird ein gleichlanges Array erzeugt, das Farbcodes für die
    Klassen der Punkte mit den entsprechenden x- und y- Koordinaten enthält.
    """
    mesh_x, mesh_y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    mesh_x = mesh_x.flatten()
    mesh_y = mesh_y.flatten()
    zz = svm_.predict(np.array([mesh_x, mesh_y]).T)
    col = [colors[c] for c in zz]
    return mesh_x, mesh_y, col


def plot_svm(svm_, x, y):
    """
    Trainiert eine SVM mit zwei Klassen und zwei Featuren und stellt das Resultat graphisch dar.
    """
    svm_.fit(x, y)
    xmin = np.min(x[:, 0])
    xmax = np.max(x[:, 0])
    ymin = np.min(x[:, 1])
    ymax = np.max(x[:, 1])
    corr = (svm_.predict(x) == y).astype(int)

    mesh_x, mesh_y, col = mesh(svm_, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    plt.scatter(mesh_x, mesh_y, c=col, marker='.')
    for label in set(y):
        plt.scatter(x[y == label, 0], x[y == label, 1], marker='+x*sDh'[label], c=['rk'[c] for c in corr[y == label]])
    if len(set(y)) <= 2:
        try:
            _a, _b, _b0, _b1 = coefs(svm_)
            plt.plot((xmin, xmax), (_b + _a * xmin, _b + _a * xmax), c='k')
            plt.plot((xmin, xmax), (_b0 + _a * xmin, _b0 + _a * xmax), 'k--')
            plt.plot((xmin, xmax), (_b1 + _a * xmin, _b1 + _a * xmax), 'k--')
        except:
            pass
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.gca().set_aspect(1)
    print(f"Score: {svm_.score(x, y)}")


# C=0 nicht erlaubt
plot_svm(SVC(kernel='linear', C=0), autos_x, autos_y)
plot_svm(SVC(kernel='linear', C=1), autos_x, autos_y)
plot_svm(SVC(kernel='linear', C=10), autos_x, autos_y)
plot_svm(SVC(kernel='linear', C=100), autos_x, autos_y)
plot_svm(SVC(kernel='linear', C=1000), autos_x, autos_y)

disk_x = 3 * np.random.random_sample((1000, 2)) - 1.5
disk_y = (np.sum(disk_x ** 2, axis=1) <= 1).astype(int)
plt.scatter(disk_x[disk_y == 1, 0], disk_x[disk_y == 1, 1], marker='+')
plt.scatter(disk_x[disk_y == 0, 0], disk_x[disk_y == 0, 1], marker='*')
plt.gca().set_aspect(1)
plot_svm(SVC(kernel='linear', C=10), disk_x, disk_y)
plot_svm(LinearSVC(C=100), disk_x, disk_y)
svm_ = SVC(kernel='linear', C=100)
svm_.fit(disk_x, disk_y)
svm_.coef_, svm_.intercept_

moons_x, moons_y = datasets.make_moons()
plot_svm(SVC(kernel='linear', C=1), moons_x, moons_y)
plot_svm(SVC(kernel='linear', C=10), moons_x, moons_y)
plot_svm(SVC(kernel='linear', C=100), moons_x, moons_y)
plot_svm(SVC(kernel='linear', C=1000), moons_x, moons_y)

plot_svm(SVC(kernel='poly', degree=2, C=1), autos_x, autos_y)
plot_svm(SVC(kernel='poly', degree=2, C=10), autos_x, autos_y)
plot_svm(SVC(kernel='poly', degree=2, C=100), autos_x, autos_y)
plot_svm(SVC(kernel='poly', degree=2, C=1000), autos_x, autos_y)
plot_svm(SVC(kernel='poly', degree=3, C=1), autos_x, autos_y)
plot_svm(SVC(kernel='poly', degree=3, C=10), autos_x, autos_y)
plot_svm(SVC(kernel='poly', degree=3, C=100), autos_x, autos_y)
plot_svm(SVC(kernel='poly', degree=3, C=1000), autos_x, autos_y)
plot_svm(SVC(kernel="poly", degree=11, C=100), moons_x, moons_y)
plot_svm(SVC(kernel="poly", degree=2, C=10), disk_x, disk_y)
plot_svm(SVC(kernel="poly", degree=2, C=100), disk_x, disk_y)
plot_svm(SVC(kernel="poly", degree=2, C=1000), disk_x, disk_y)
plot_svm(SVC(kernel="poly", degree=3, C=1), disk_x, disk_y)
plot_svm(SVC(kernel="poly", degree=3, C=100), disk_x, disk_y)
plot_svm(SVC(kernel="poly", degree=4, C=100), disk_x, disk_y)
rbf - Kernel(Gauß - Kernel)(Default)
plot_svm(SVC(kernel="rbf", C=1), disk_x, disk_y)
plot_svm(SVC(kernel="rbf", C=10), disk_x, disk_y)
plot_svm(SVC(kernel="rbf", C=100), disk_x, disk_y)
plot_svm(SVC(kernel="rbf", C=1000), disk_x, disk_y)
plot_svm(SVC(kernel="rbf", gamma=1, C=100), disk_x, disk_y)
plot_svm(SVC(kernel="rbf", gamma=0.01, C=100), disk_x, disk_y)
plot_svm(SVC(kernel="rbf", C=1), moons_x, moons_y)
plot_svm(SVC(kernel="rbf", C=1000), moons_x, moons_y)
Mehrklassen - Probleme
autos2 = pd.read_csv("Auto2MerkmaleClass.csv", header=None)
autos2.head()
autos2_x = scaler.fit_transform(autos2[[1, 2]])
autos2_y = autos2[0].values
plot_svm(SVC(kernel='linear', C=10), autos2_x, autos2_y)
plot_svm(SVC(kernel='linear', C=1000), autos2_x, autos2_y)
plot_svm(SVC(kernel='rbf', C=1000), autos2_x, autos2_y)
plot_svm(SVC(kernel='rbf', gamma=1000, C=1000), autos2_x, autos2_y)
MNist
Digits - dataset
von
sklearn
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.8)
svm_d = SVC(kernel="linear", C=1)
svm_d.fit(X_train, y_train)
svm_d.score(X_test, y_test)
svm_d = SVC(kernel="linear", C=10)
svm_d.fit(X_train, y_train)
svm_d.score(X_test, y_test)
svm_d = SVC(kernel="linear", C=100)
svm_d.fit(X_train, y_train)
svm_d.score(X_test, y_test)
svm_d = SVC(kernel="linear", C=1000)
svm_d.fit(X_train, y_train)
svm_d.score(X_test, y_test)
svm_d = LinearSVC(C=10)
svm_d.fit(X_train, y_train)
svm_d.score(X_test, y_test)

mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")
mnist_train.shape, mnist_test.shape
mnist_train_x = mnist_train.iloc[:, 1:]
mnist_train_y = mnist_train.label
mnist_test_x = mnist_test.iloc[:, 1:]
mnist_test_y = mnist_test.label
svm_k = SVC(kernel="linear", C=1)
svm_k.fit(mnist_train_x, mnist_train_y)
svm_k.score(mnist_test_x, mnist_test_y)
svm_k = LinearSVC(C=10)
svm_k.fit(mnist_train_x, mnist_train_y)
svm_k.score(mnist_test_x, mnist_test_y)
svm_k = SGDClassifier()
svm_k.fit(mnist_train_x, mnist_train_y)
svm_k.score(mnist_test_x, mnist_test_y)