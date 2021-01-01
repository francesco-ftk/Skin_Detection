from test_and_training_set import save_frame_from_video, create_training_and_test_set
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from table_plot_functions import printTable, printConfusionMatrix

precision=[]
recall=[]
accuracy=[]
classifier=[]

# Salvataggio frame da video
totalFrame=save_frame_from_video()

# Creazione Training Set e Test Set
Xtraining, Xtest, ytraining, ytrue= create_training_and_test_set(totalFrame)
print("#samples: ", len(Xtraining), "#features: ", 3)

# Classificatore Random Forest
print("Random_Forest_Classifier: ")
clf=RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
clf.fit(Xtraining,ytraining)
print("Classificatore Random Forest addestrato")
prediction=clf.predict(Xtest)
print("Random_Forest Contingency Matrix")
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

# Classificatore Gaussian Naive Bayes
print("Gaussian_Naive_Bayes_Classifier: ")
clf=GaussianNB()
clf.fit(Xtraining,ytraining)
print("Classificatore Gaussian Naive Bayes addestrato")
prediction=clf.predict(Xtest)
print("Gaussian_Naive_Bayes Contingency Matrix")
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

# Stampa risultati in tabella
printTable(classifier,precision,recall,accuracy)



