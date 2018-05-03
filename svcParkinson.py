from scipy.io import loadmat, savemat
from scipy import stats
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import math
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report,accuracy_score

x = loadmat('/media/yasin/912d5d0f-5c5b-494b-bb8c-245aed208848/home/yasin/washingtonData/parkinsonFeatures/parkinsonClassificationFeaturesNormal.mat')
feats = x['features']

x = loadmat('/media/yasin/912d5d0f-5c5b-494b-bb8c-245aed208848/home/yasin/washingtonData/parkinsonFeatures/parkinsonClassificationLabels.mat')
labs = (x['labels'])

x = loadmat('/media/yasin/912d5d0f-5c5b-494b-bb8c-245aed208848/home/yasin/washingtonData/parkinsonFeatures/mrmrForParkinsonClassificationFeatures.mat')
selFeat = (x['featFCD'])

allPred = []
numFeat = [3,4,5,10,15,20,40,80,100,200,400,800,1000,1200,1500,2000,2200]

trainFeats, devFeats, trainLabs, devLabs = train_test_split(
     feats, labs, test_size=0.1, random_state=0)

selFeat -= 1

for j in range(0, len(numFeat)):


    selFeat2 = selFeat[:, 0:numFeat[j]]

    trainFeats2 = trainFeats[:, selFeat2[0, :]]
    devFeats2 = devFeats[:, selFeat2[0, :]]





    #clf = SVC(C=20, kernel='rbf')
    #clf = ExtraTreeClassifier(random_state=100)
    #clf = DecisionTreeClassifier(splitter='best',criterion='gini',random_state=100)
    clf = RandomForestClassifier(criterion='gini', random_state=100, n_estimators = 200,verbose=1)
    #clf = MLPClassifier(hidden_layer_sizes=(512,3))
    clf.fit(trainFeats2, trainLabs.ravel())
    predictions = clf.predict(devFeats2)

    print("The method is: Random Forest n_estimators = 200")
    print('*****************************************************************')
    print(classification_report(devLabs,predictions))
    print("Accuracy: ",accuracy_score(devLabs,predictions))
    print("Number of selected features: ",numFeat[j])
    print('******************************************************************')
    allPred.clear()
