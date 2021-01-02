from test_and_training_set import save_frame_from_video, create_training_and_test_set
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from table_plot_functions import printTable, printConfusionMatrix
from plot_functions import graphic_plot, histogram_plot

precision=[]
recall=[]
accuracy=[]
classifier=[]
f_score_histogram=[]
f_score=[]

# Salvataggio frame da video
totalFrame=save_frame_from_video()

# Creazione Training Set e Test Set
Xtraining, Xtest, ytraining, ytrue= create_training_and_test_set(totalFrame)
print("#samples: ", len(Xtraining), "#features: ", 3)

# Classificatore Gaussian Naive Bayes
print("Gaussian_Naive_Bayes_Classifier... ")
clf=GaussianNB()
clf.fit(Xtraining,ytraining)
print("Training finished")
f_score_histogram.append(round(clf.score(Xtest,ytrue),2))
prediction=clf.predict(Xtest)
# Gaussian_Naive_Bayes Contingency Matrix
tn, fp, fn, tp = confusion_matrix(ytrue, prediction).ravel()
printConfusionMatrix(tn, fp, fn, tp)
p=tp/(tp+fp)
r=tp/(tp+fn)
a=(tp+tn)/(tp+fp+fn+tn)
p=round(p,3)
r=round(r,3)
a=round(a,3)
precision.append(p)
recall.append(r)
accuracy.append(a)
classifier.append("Gaussian_Naive_Bayes")

# Classificatore Random Forest
n_estimators=[1,4,7,10,11]
for i in range(len(n_estimators)):
    print("Random_Forest_Classifier, #estimators: %d" % n_estimators[i])
    clf=RandomForestClassifier(n_estimators=n_estimators[i], criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
    clf.fit(Xtraining,ytraining)
    print("Training completed")
    fscore=round(clf.score(Xtest,ytrue),2)
    f_score.append(fscore)
    if n_estimators[i]==10:
        f_score_histogram.append(fscore)
        prediction=clf.predict(Xtest)
        #Random_Forest Contingency Matrix
        tn, fp, fn, tp = confusion_matrix(ytrue, prediction).ravel()
        printConfusionMatrix(tn, fp, fn, tp)
        p=tp/(tp+fp)
        r=tp/(tp+fn)
        a=(tp+tn)/(tp+fp+fn+tn)
        p=round(p,3)
        r=round(r,3)
        a=round(a,3)
        precision.append(p)
        recall.append(r)
        accuracy.append(a)
        classifier.append("Random_Forest")

# Stampa risultati
histogram_plot(f_score_histogram)
graphic_plot(n_estimators,f_score)
printTable(classifier,precision,recall,accuracy)


