import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def pre_process(df: pd.DataFrame):
    race_mapper = {"1. White": 0, "2. Black": 1, "3. Asian": 2, "4. Other": 3}
    # education_mapper = {"1. < HS Grad": 0, "2. HS Grad": 1, "3. Some College": 2, "4. College Grad": 3, "5. Advanced Degree": 4}
    jobclass_mapper = {"1. Industrial": 0, "2. Information": 1}
    health_mapper = {"1. <=Good": 0, "2. >=Very Good": 1}
    health_ins_mapper = {"1. Yes": 0, "2. No": 1}

    df = pd.get_dummies(df, columns=["maritl", "education"])

    df["race"].replace(race_mapper, inplace=True)
    # df["education"].replace(education_mapper, inplace=True)
    df["jobclass"].replace(jobclass_mapper, inplace=True)
    df["health"].replace(health_mapper, inplace=True)
    df["health_ins"].replace(health_ins_mapper, inplace=True)

    return df


def main():
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    train_data.dropna(inplace=True)

    train_data = pre_process(train_data)
    test_data = pre_process(test_data)

    train_X = train_data.loc[:, train_data.columns != "race"]
    train_Y = train_data.iloc[:, 2]

    test_X = test_data.loc[:, test_data.columns != "race"]
    test_Y = test_data.iloc[:, 2]

    pca = PCA(n_components=4)
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)

    ada = AdaBoostClassifier(n_estimators=50, learning_rate=0.01, random_state=3427, base_estimator=DecisionTreeClassifier(max_depth=15))
    ada.fit(train_X, train_Y)
    predict_Y = ada.predict(test_X)
    
    print(f1_score(test_Y, predict_Y, average='macro'))


if __name__ == '__main__':
    main()
