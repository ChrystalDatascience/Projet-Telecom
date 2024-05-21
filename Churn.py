"""
This churn_library.py consists of creating a
Machine Learning model that can predict whether 
a customer will unsubscribe or not.
    Directed by: Chrystal Orian VIGAN
    Date: 20/05/2024
"""

# Importation des Librairies et Packages

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay

import joblib

# Pour la Journalisation
logging.basicConfig(
    filename="./logs/churn.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

# Importation des données
def import_path(path):
    """
    retourner un dataframe en prenant le chemin du dataset

    input:
        path: le chemin
    output:
        return un dataframe
    """
    Data = pd.read_csv(path)
    data = Data.copy()
    data.drop('customerID', axis=1, inplace=True)
    return data

def convert_total_charge(data):
    """
    retournera le dataframe en remplaçant les " " de la colonne TotalCharges par des valeurs manquante
    input:
        data: dataframe
    output:
        dataframe: dataframe avec la colonne TotalCharge modifié
    """
    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan).astype(float)
    return data

def split(data):
    """
    retournera le dataframe en un trainset, tastset et valset

    input:
        data:dataframe
    output:
        tainset: 70% de data
        testset: 15% de data
        valset: 15% de data
    """
    trainset, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Churn'])
    testset, valset = train_test_split(test, train_size=0.5, random_state=0, stratify=test['Churn'])

    # Enregistrement des jeux de données
    trainset.to_csv("./data/train.csv", index=False)
    testset.to_csv('./data/test.csv', index=False)
    valset.to_csv('./data/validate.csv', index=False)

    return trainset, valset

def cut_x_y(trainset, valset):
    """retournera les features et les targets

    input: trainset, valset
    output: x_train, x_val, y_train, y_val
    """
    trainset['Churn'] = trainset['Churn'].map({'No':0, 'Yes':1})
    valset['Churn'] = valset['Churn'].map({'No':0, 'Yes':1})

    x_train, y_train = trainset.drop('Churn', axis=1), trainset['Churn']
    x_val, y_val = valset.drop('Churn', axis=1), valset['Churn']

    return x_train, y_train, x_val, y_val

def eda(trainset):
    """ Enregistre les figures dans le dossier Image

    input:
        data: trainset
    output:
        None
    """
    df = trainset.copy()
    for col in df.select_dtypes(include='object').columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(data=df, x=col)
        plt.title('Diagramme à barre de: ' +col)
        plt.savefig("Images/" + col + ".jpg") 

        plt.figure(figsize=(12, 8))
        sns.heatmap(pd.crosstab(df['Churn'], df[col]), annot=True, fmt='d')
        plt.savefig("Images/" + col + ".jpg") 
        plt.close()
    
    
    num = df.select_dtypes(exclude='object')
    plt.figure(figsize=(12, 8))
    sns.heatmap(num.corr(), annot=True, cmap="RdBu", fmt='2g')
    plt.savefig("Images/Correlation_variable_numerique.jpg")

    for col in df.select_dtypes(exclude='object').columns:
        plt.figure(figsize=(12, 8))
        df[col].hist()
        plt.savefig("Images/" + col + ".jpg") 
        plt.close()

def classi_report(y_train, y_train_pred,
                                y_val, y_val_pred):
    '''
    Produira les rapports de classification pour les données du train et du val
    
    input:
            y_train: target du trainset
            y_train_preds: prediction sur les données du trainavec le model de regression logistic
            y_val: target du valset
            y_val_preds: prediction sur les données du valset avec le model de regression logistic
    output:
            None
    '''
    class_report_dico = {
        "Logistique Regression train results": classification_report(
            y_train,
            y_train_pred
        ),
         "Logistic Regression validation result": classification_report(
            y_val,
            y_val_pred
        )
    }
    
    for title, report in class_report_dico.items():
        plt.rc('figure', figsize=(7, 3))
        plt.text(
           0.2, 0.3, str(report), {
               'fontsize': 10
           },
           fontproperties='monospace'
        )
        plt.axis('off')
        plt.title(title, fontweight='bold')
        plt.savefig("Images/results/" + title + ".jpg")
        plt.close()

def pipeline_modeli():

    numerical_df = make_column_selector(dtype_include=np.number)
    categorical_df = make_column_selector(dtype_exclude=np.number)  

    # Pileline de preprocessing
    numerical_transformer = Pipeline(
        steps= [
            ('imputer', SimpleImputer(strategy='most_frequent')),
                ('standard', StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('encoder', OneHotEncoder())
        ]
    )
    
    # Combinaison des deux pipeline
    preprocessing = ColumnTransformer(
        transformers= [
            ('numeric', numerical_transformer, numerical_df),
            ('categorial', categorical_transformer, categorical_df)
        ]
    )
    
    # Pipeline de modelisation
    model = make_pipeline(
        preprocessing, LogisticRegression(random_state=0, max_iter=1000, C= 0.5, penalty= None, solver= 'lbfgs' )
    )

    return model

def pred(x_train, x_val, y_train):
    model = pipeline_modeli()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    return y_train_pred, y_val_pred


def training(x_train, x_val, y_train, y_val):
    """Entrainera le model sur les données du trainset et affichera la courbe d'apprentissage

    input:
        x_train : x train data
        x_val : x valset data
        y_train : Y train data
        y_val : Y valset data
    output:
        None
    """
    # Entrainement
    model = pipeline_modeli()
    model.fit(x_train, y_train)

    # Prediction
    y_val_pred = model.predict(x_val)

    

    # matrix de confusion
    print(confusion_matrix(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    #learning curve
    N, train_score, val_score = learning_curve(model , x_train, y_train, cv=4,
                                            train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title('Regression Logistic')
    plt.legend()
    plt.savefig("Images/results/Regression Logistic.jpg")

    #Sauvegarde du model
    joblib.dump(model, './models/logreg_model.pkl')



def main():

    logging.info("Importation des données... ")
    data = import_path("./data/WA_Fn-UseC_-Telco-customer-Churn.csv")
    logging.info("Données importer avec succès.")

    logging.info("Convertion TotalCharges... ")
    df = convert_total_charge(data)
    logging.info("Convertion TotalChages: Succès")

    logging.info("Division des Données... ")
    trainset, valset = split(df)
    logging.info("Division des données: Succès")
    
    logging.info("Découpage des données en x et y... ")
    x_train, y_train, x_val, y_val = cut_x_y(trainset, valset)
    logging.info("Découpage des données en x et y avec succès")

    logging.info("Analyse exploratoire des données... ")
    eda(trainset)
    logging.info("Analyse exploratoire des données: Succès")

    logging.info("Formation du model... ")
    training(x_train, x_val, y_train, y_val)
    logging.info("Formation du model: Succès")

    logging.info("Prediction en cours...")
    y_train_pred, y_val_pred = pred(x_train, x_val, y_train)
    logging.info("Prediction en cours: Succès")

    logging.info("Rapport de classification... ")
    classi_report(y_train, y_train_pred,y_val, y_val_pred)
    logging.info("Rapport de classification: Succès ")

if __name__ == "__main__":
    print("Execution en cours...")
    main()
    print("Fin de l'execution")