import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles

# Construct train dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=1000, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=1500, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
Xtrain,Xvalid,ytrain,yvalid=train_test_split(X,y,test_size=0.2)

fig, axs = plt.subplots(3,3)

axs[0,0].scatter(Xtrain[ytrain==0,0],Xtrain[ytrain==0,1],c='b', s=0.1)
axs[0,0].scatter(Xtrain[ytrain==1,0],Xtrain[ytrain==1,1],c='r', s=0.1)


#-------------------------------------------------------
def plotDecisionBoundaries(bagging_or_boosting, n_estimators, Xtrain, ytrain, subplot, plot_step=0.02,
                           plot_colors="br", class_names="AB", max_depth=1):
    if bagging_or_boosting == 'boosting':
        # Create and fit an AdaBoosted decision tree
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                                 algorithm="SAMME",
                                 n_estimators=n_estimators)
    elif bagging_or_boosting == 'bagging':
        # Create and fit an bagged decision tree
        bdt = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth),
                                n_estimators=n_estimators)

    bdt.fit(Xtrain, ytrain)

    # Plot the decision boundaries
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = subplot.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    subplot.axis("tight")

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        subplot.scatter(Xtrain[ytrain == i, 0], Xtrain[ytrain == i, 1],
                    c=c, s=3, edgecolors='none',
                    label="Class %s" % n)
    subplot.set_xlim(x_min, x_max)
    subplot.set_ylim(y_min, y_max)
    subplot.legend(loc='upper right')
    subplot.set_xlabel('x')
    subplot.set_ylabel('y')
    subplot.set_title('Decision Boundary, {0} decision trees of depth 1'.format(bagging_or_boosting))

    return bdt
#--------------------------------------------------------------

n_estimators=100
plotDecisionBoundaries('bagging',n_estimators,Xtrain,ytrain, axs[0,1])

n_estimators=100
plotDecisionBoundaries('boosting',n_estimators,Xtrain,ytrain, axs[0,2])

trainErrorList = []
validationErrorList = []
max_depth = 2

n_estimator_list = range(10, 500, 50)
for n_estimators in n_estimator_list:
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                             algorithm="SAMME",
                             n_estimators=n_estimators)

    bdt.fit(Xtrain, ytrain)

    yhattrain = bdt.predict(Xtrain)
    yhat_val = bdt.predict(Xvalid)

    train_error = 1 - accuracy_score(yhattrain, ytrain)
    val_error = 1 - accuracy_score(yhat_val, yvalid)

    trainErrorList.append(train_error)
    validationErrorList.append(val_error)

axs[1,0].plot(n_estimator_list,trainErrorList,'b',label='training error')
axs[1,0].plot(n_estimator_list,validationErrorList,'r',label='validation error')
axs[1,0].legend()

estimator=bdt.estimators_[0]
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
print(n_nodes,children_left,children_right,feature,threshold)

n_estimators=50 #Versuchen Sie auch 500
bdt=plotDecisionBoundaries('boosting',n_estimators,Xtrain,ytrain,axs[1,1])
for est in bdt.estimators_:
    th = est.tree_.threshold[0]
    if est.tree_.feature[0]==0:
        # print('x-th {0:+1.2f}'.format(th))
        axs[1,1].plot([th,th],plt.ylim(),'b')
    else:
        # print('y-th {0:+1.2f}'.format(th))
        axs[1,1].plot(plt.xlim(),[th,th],'r')


n_estimators=50

plotDecisionBoundaries('boosting',n_estimators,Xtrain,ytrain,axs[1,2])

plotDecisionBoundaries('bagging',n_estimators,Xtrain,ytrain,axs[2,0])

plt.show()