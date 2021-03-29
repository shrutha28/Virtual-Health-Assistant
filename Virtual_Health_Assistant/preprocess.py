import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def randomforest(sym1,sym2,sym3,sym4):
    model = pickle.load(open('model/RFC_model.sav', 'rb'))
    df2=pd.read_csv("vha.csv",index_col=[0])
    y = df2["Disease"]
    multi = MultiColumnLabelEncoder().fit_transform(df2.drop('Disease',axis=True))
    X = multi

    #x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)
    #model = RandomForestClassifier(n_estimators = 200,criterion="entropy")
    #model.fit(x_train,y_train)
    #y_pred=model.predict(x_test)
    #acc = accuracy_score(y_test,y_pred)
    #train_acc = accuracy_score(y_train,model.predict(x_train))

    sym = [sym1, sym2, sym3, sym4]
    test_list = ' '.join(sym).split()
    for i in range(4-len(test_list)):
        test_list.append('None')
    df3=df2.drop('Disease',axis=True)
    df3.loc[4920] = test_list
    multi1 = MultiColumnLabelEncoder().fit_transform(df3)
    arr=multi1.loc[4920].to_numpy()
    #pred
    predict = model.predict(arr.reshape(1,-1))
    return predict
    #print(predict)
