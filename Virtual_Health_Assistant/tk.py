

from tkinter import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
import pickle

df = pd.read_csv("dataset.csv")
#df
df.isnull().sum()
df = df.drop(["Symptom_5","Symptom_6","Symptom_7","Symptom_8","Symptom_9","Symptom_10","Symptom_11","Symptom_12","Symptom_13","Symptom_14","Symptom_15", "Symptom_16","Symptom_17"],axis=1)
df.fillna("None",inplace=True)

symptoms =["Symptom_1","Symptom_2","Symptom_3","Symptom_4"]

for i in symptoms:
    df[i]=df[i].str.strip()

df['Disease']=df['Disease'].str.strip()

A=list(df["Symptom_1"].unique())
B=list(df["Symptom_2"].unique())
C=list(df["Symptom_3"].unique())
D=list(df["Symptom_4"].unique())

s = list(set(A+B+C+D))
s=sorted(s)

"""
for i in symptoms:
    df.replace({i:{'abdominal_pain':0, 'acidity':1, 'altered_sensorium':2, 'anxiety':3, 'back_pain':4, 'blackheads':5, 'bladder_discomfort':6, 
             'blister':7, 'bloody_stool':8, 'blurred_and_distorted_vision':9, 'breathlessness':10, 'bruising':11, 'burning_micturition':12, 
             'chest_pain':80, 'chills':81, 'cold_hands_and_feets':82, 'constipation':83, 'continuous_feel_of_urine':84, 'continuous_sneezing':85, 
             'cough':13, 'cramps':14, 'dark_urine':15, 'dehydration':16, 'diarrhoea':17, 'dischromic _patches':18, 'distention_of_abdomen':19, 'dizziness':20, 
             'excessive_hunger':21, 'extra_marital_contacts':22, 'family_history':23, 'fatigue':24, 'foul_smell_of urine':25, 'headache':26, 'high_fever':27, 
             'hip_joint_pain':28, 'indigestion':29, 'irregular_sugar_level':30, 'irritation_in_anus':31, 'itching':32, 'joint_pain':33, 'knee_pain':34, 
             'lack_of_concentration':35, 'lethargy':36, 'loss_of_appetite':37, 'loss_of_balance':38, 'mood_swings':39, 'movement_stiffness':40, 'muscle_wasting':41, 
             'muscle_weakness':42, 'nausea':43, 'neck_pain':44, 'nodal_skin_eruptions':45, 'obesity':46, 'pain_during_bowel_movements':47, 'pain_in_anal_region':48, 
             'painful_walking':49, 'passage_of_gases':50, 'patches_in_throat':51, 'pus_filled_pimples':52, 'red_sore_around_nose':53, 'restlessness':54, 'scurring':55, 
             'shivering':56, 'silver_like_dusting':57, 'skin_peeling':58, 'skin_rash':59, 'small_dents_in_nails':60, 'spinning_movements':61, 'spotting_ urination':62, 
             'stiff_neck':63, 'stomach_pain':64, 'sunken_eyes':65, 'sweating':66, 'swelling_joints':67, 'swelling_of_stomach':68, 'swollen_legs':69, 'ulcers_on_tongue':70, 
             'vomiting':71, 'watering_from_eyes':72, 'weakness_in_limbs':73, 'weakness_of_one_body_side':74, 'weight_gain':75, 'weight_loss':76, 'yellow_crust_ooze':77, 
             'yellowing_of_eyes':78, 'yellowish_skin':79,'None':86}},inplace=True)
"""
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




multi = MultiColumnLabelEncoder().fit_transform(df.drop('Disease',axis=1))
#multi
 
y = df["Disease"]
X = multi

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0) 


#200, RFC MODEL     
model = RandomForestClassifier(n_estimators = 200,criterion="entropy")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
train_acc = accuracy_score(y_train,model.predict(x_train))
print(acc)
print(train_acc)


filename = 'RFC_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
#print(result)


def randomforest():
    model = RandomForestClassifier(n_estimators = 200,criterion="entropy")
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    train_acc = accuracy_score(y_train,model.predict(x_train))
    print(acc)
    print(train_acc)
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get()]
    sym = []
    for k in psymptoms:
        sym.append(k)
    
    df2=df.drop('Disease',axis=True)    
    #df2
    df2.loc[4920] = sym
    multi = MultiColumnLabelEncoder().fit_transform(df2)
    arr=multi.loc[4920].to_numpy()
    #pred
    predict = model.predict(arr.reshape(1,-1))
    print(predict)
    t3.insert(END,predict)


#GUI
root = Tk()
root.configure(background='blue')

# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Name = StringVar()

# Heading
w2 = Label(root, justify=LEFT, text="Disease Predictor using Machine Learning", fg="white", bg="blue")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2 = Label(root, justify=LEFT, text="A Project by Sruthi", fg="white", bg="blue")
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15, sticky=W)


S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)


destreeLb = Label(root, text="RandomForest", fg="white", bg="red")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)


# entries
OPTIONS = s

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)


rnf = Button(root, text="Randomforest", command=randomforest,bg="green",fg="black")
rnf.grid(row=9, column=3,padx=10)



#textfileds
t3 = Text(root, height=1, width=40,bg="orange",fg="black")
t3.grid(row=19, column=1 , padx=10)

root.mainloop()

