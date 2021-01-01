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
n_estimators=[]

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
print("Random_Forest_Classifier with one estimator... ")
clf=RandomForestClassifier(n_estimators=1, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
clf.fit(Xtraining,ytraining)
print("Training finished")
f_score.append(round(clf.score(Xtest,ytrue),2))
n_estimators.append(1)

print("Random_Forest_Classifier with four estimators... ")
clf=RandomForestClassifier(n_estimators=4, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
clf.fit(Xtraining,ytraining)
print("Training finished")
f_score.append(round(clf.score(Xtest,ytrue),2))
n_estimators.append(4)

print("Random_Forest_Classifier with seven estimators... ")
clf=RandomForestClassifier(n_estimators=7, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
clf.fit(Xtraining,ytraining)
print("Training finished")
f_score.append(round(clf.score(Xtest,ytrue),2))
n_estimators.append(7)

print("Random_Forest_Classifier with ten estimators... ")
clf=RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
clf.fit(Xtraining,ytraining)
print("Training finished")
fscore=round(clf.score(Xtest,ytrue),2)
f_score_histogram.append(fscore)
f_score.append(f_score)
n_estimators.append(10)
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

print("Random_Forest_Classifier with twelve estimators... ")
clf=RandomForestClassifier(n_estimators=12, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
clf.fit(Xtraining,ytraining)
print("Training finished")
f_score.append(round(clf.score(Xtest,ytrue),2))
n_estimators.append(12)

# Stampa risultati
graphic_plot(n_estimators,f_score)
printTable(classifier,precision,recall,accuracy)



