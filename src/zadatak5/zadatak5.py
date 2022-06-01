import sys
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score


def pre_process(df: pd.DataFrame):
    region_mapper = {"Africa": 0, "Asia": 1, "Americas": 2, "Europe": 3}
    oil_mapper = {"yes": 1, "no": 0}
    
    df["oil"].replace(oil_mapper, inplace=True)
    df["region"].replace(region_mapper, inplace=True)

    return df


def main():
    # train_file_path = sys.argv[1]
    # test_file_path = sys.argv[2]

    train_file_path = "C:\\Users\\Dusan\\Documents\\github repos\\venice_and_genoa\\src\\zadatak5\\res\\train.csv"
    test_file_path = "C:\\Users\\Dusan\\Documents\\github repos\\venice_and_genoa\\src\\zadatak5\\res\\test.csv"

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    train_data.dropna(inplace=True)

    train_data = pre_process(train_data)
    test_data = pre_process(test_data)

    train_data.query("income != 3010", inplace=True) # outliers in africa
    train_data.query("income != 1000", inplace=True) # outliers in africa
    train_data.query("income != 5523", inplace=True) # outliers in america
    train_data.query("income != 4751", inplace=True) # outliers in america
    train_data.query("income != 2526", inplace=True) # outliers in asia
    train_data.query("income != 1530", inplace=True) # outliers in asia
    train_data.query("income != 406 or region != 3", inplace=True) # outliers in europe


    train_X = train_data.iloc[:, [0, 1, 3]]
    train_Y = train_data.iloc[:, 2]

    test_X = test_data.iloc[:, [0, 1, 3]]
    test_Y = test_data.iloc[:, 2]

    mix = GaussianMixture(n_components=4, random_state=38953, covariance_type="tied", init_params="random")

    mix.fit(train_X)
    predict_Y = mix.predict(test_X)
    
    print(v_measure_score(test_Y, predict_Y))


if __name__ == '__main__':
    main()
