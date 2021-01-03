from test_and_training_set import save_frame_from_video, create_training_and_test_set
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from table_plot_functions import printTable, printConfusionMatrix, histogram_plot

precision=[]
recall=[]
accuracy=[]
classifier=[]
f_score=[]

numeroVideo=input("Inserire il numero di video grezzi da elaborare: ")
# Salvataggio frame da video
totalFrame=save_frame_from_video(int(numeroVideo))

# Creazione Training Set e Test Set
Xtraining, Xtest, ytraining, ytrue= create_training_and_test_set(totalFrame)
print("#samples: ", len(Xtraining), "#features: ", 3)

# Classificatore Gaussian Naive Bayes
print("Gaussian_Naive_Bayes_Classifier")
clf=GaussianNB()
clf.fit(Xtraining,ytraining)
print("Training finished")
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
# F-score
fscore=(2*(p*r)/(p+r))
f_score.append(round(fscore,3))

# Classificatore Random Forest
print("Random_Forest_Classifier")
clf=RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=True, max_samples=None)
clf.fit(Xtraining,ytraining)
print("Training completed")
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
# F-score
fscore=(2*(p*r)/(p+r))
f_score.append(round(fscore,3))

# Stampa risultati
histogram_plot(f_score)
printTable(classifier,precision,recall,accuracy)


