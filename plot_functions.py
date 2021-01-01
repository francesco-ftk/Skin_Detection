import matplotlib.pyplot as plt
import numpy as np

def graphic_plot(n_estimators,f_score):
    plt.plot(n_estimators,f_score,color="blue", marker="o",linestyle="-")
    plt.ylabel('F-score')
    plt.xlabel('n_estimators')
    plt.title('F-score vs number of trees grown for Random Forest')
    plt.show()

def histogram_plot(f_score):
    x = [1,2]
    plt.bar(x, height=f_score)
    plt.ylabel('F-score')
    plt.xticks(x,["Gaussian_Naive_Bayes","Random_Forest"])
    plt.show()

