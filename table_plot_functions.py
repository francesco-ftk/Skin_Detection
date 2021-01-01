import plotly.graph_objects as go

def printTable(classifier,precision,recall,accuracy):
    table= go.Figure(data=[go.Table(header=dict(values=['Classifier','Precision','Recall','Accuracy'],align=['left'],font=dict(size=13)),
    cells=dict(values=[classifier,precision,recall,accuracy],align='left',font_size=12))])
    table.show()

def printConfusionMatrix(tn, fp, fn, tp):
    table= go.Figure(data=[go.Table(header=dict(values=['Prediction/True_Class','Skin','Background'],align=['center'],font=dict(size=13)),
    cells=dict(values=[['Skin','Background'],[tp,fn],[fp,tn]],align='center',font_size=13))])
    table.show()



