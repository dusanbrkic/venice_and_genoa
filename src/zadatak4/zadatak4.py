import sys
from turtle import speed
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def map(df):
    speed_mapper = {"1-9km/h": 0, "10-24": 1, "25-39": 2, "40-54": 3, "55+": 5}
    dead_mapper = {"dead": 0, "alive": 1}
    airbag_mapper = {"none": 0, "airbag": 1}
    seatbelt_mapper = {"none": 0, "belted": 1}
    sex_mapper = {"m": 0, "f": 1}
    #abcat_mapper = {"deploy":0, "nondeploy":1, "unavail":2}
    role_mapper = {"driver": 0, "pass": 1}

    abcat = pd.get_dummies(df["abcat"])
    del df["deploy"]
    del df["abcat"]
    df["speed"].replace(speed_mapper, inplace=True)
    df["dead"].replace(dead_mapper, inplace=True)
    df["airbag"].replace(airbag_mapper, inplace=True)
    df["seatbelt"].replace(seatbelt_mapper, inplace=True)
    df["sex"].replace(sex_mapper, inplace=True)
    df["occRole"].replace(role_mapper, inplace=True)
    df = df.join(abcat)
    return df


def main():
    #train_file_path = sys.argv[1]
    #test_file_path = sys.argv[2]

    train_file_path = "G:\\SE\\Git\\venice_and_genoa\\src\\zadatak4\\res\\train.csv"
    test_file_path = "G:\\SE\\Git\\venice_and_genoa\\src\\zadatak4\\res\\test_preview.csv"

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    train_data = train_data.replace(to_replace='None', value=np.nan).dropna()

    train_data = map(train_data)
    test_data = map(test_data)

    train_X = train_data.iloc[:, 1:]
    train_Y = train_data.iloc[:, :1]

    test_X = test_data.iloc[:, 1:]
    test_Y = test_data.iloc[:, :1]

    rf = RandomForestClassifier(n_estimators=10)

    #lr = LogisticRegression()
    #dt = DecisionTreeClassifier()
    #svm = SVC(kernel="poly", degree=2)

    # evc = VotingClassifier(
    #    estimators=[("lr", lr), ("dt", dt), ("svm", svm)], voting="hard")
    #evc.fit(train_X, train_Y)

    rf.fit(train_X, train_Y)
    predict_Y = rf.predict(test_X)

    #predict_Y = evc.predict(test_X)
    print(classification_report(predict_Y, test_Y))


if __name__ == '__main__':
    main()
