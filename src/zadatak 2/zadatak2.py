import sys
import numpy as np
import pandas
from matplotlib import pyplot as plt


class Ridge:
    def __init__(self, alfa=1):
        self.alfa = alfa
        self.koeficijenti = None

    def treniraj(self, X, y):
        dim = X.shape
        jedinicna = np.identity(dim[1])

        self.koeficijenti = np.dot(np.dot(np.linalg.inv(
            np.dot(X.T, X) + self.alfa * jedinicna), X.T), y)
        return

    def predvidi(self, parametri):
        # print(self.koeficijenti)
        y = [0 for o in range(len(parametri))]
        for j in range(len(parametri)):
            y[j] = self.koeficijenti[0]
            for i in range(1, len(self.koeficijenti)):
                y[j] = y[j] + parametri[j][i] * self.koeficijenti[i]
        return y


def dodaj_jedinice(X):
    dim = X.shape
    jedinice = np.full((dim[0], 1), 1)
    X = np.c_[jedinice, X]
    return X


def stepenuj_matricu(X, stepen):
    Y = X
    # print(X)
    for i in range(1, stepen):
        X = np.c_[X, Y ** (i + 1)]
    # print(X)
    return X


def rmse(true_y, pred_y):
    return np.sqrt(np.mean((pred_y - true_y) ** 2))


def main():
    #train_file_path = sys.argv[1]
    train_file_path = "G:\\SE\\Git\\venice_and_genoa\\src\\zadatak 2\\res\\train.csv"
    #test_file_path = sys.argv[2]
    test_file_path = "G:\\SE\\Git\\venice_and_genoa\\src\\zadatak 2\\res\\test_set.csv"
    print(sys.argv)
    train_data = pandas.read_csv(train_file_path)
    test_data = pandas.read_csv(test_file_path)

    zvanja = pandas.get_dummies(train_data["zvanje"])
    oblasti = pandas.get_dummies(train_data["oblast"])
    train_data.drop(["zvanje", "oblast", "pol"], inplace=True, axis=1)
    train_data = train_data.join(zvanja)
    train_data.drop(["AsstProf"], inplace=True, axis=1)
    train_data = train_data.join(oblasti["A"])

    zvanja_test = pandas.get_dummies(test_data["zvanje"])
    oblasti_test = pandas.get_dummies(test_data["oblast"])
    test_data.drop(["zvanje", "oblast", "pol"], inplace=True, axis=1)
    test_data = test_data.join(zvanja_test)
    test_data.drop(["AsstProf"], inplace=True, axis=1)
    test_data = test_data.join(oblasti_test["A"])

    plate = train_data["plata"]
    train_data.drop(["plata"], inplace=True, axis=1)

    plate_test = test_data["plata"]
    test_data.drop(["plata"], inplace=True, axis=1)

    train_X = train_data
    train_y = plate
    test_X = test_data.to_numpy()
    test_y = plate_test.to_numpy()

    test_X = stepenuj_matricu(test_X, 1)
    test_X = dodaj_jedinice(test_X)

    np.set_printoptions(threshold=np.inf)
    regression = Ridge(0.001389)
    train_X = stepenuj_matricu(train_X.to_numpy(), 1)
    train_X = dodaj_jedinice(train_X)
    regression.treniraj(train_X, train_y)

    # print(test_X)

    pred_Y = regression.predvidi(test_X)

    print(rmse(test_y, pred_Y))

    # print(train_X)
    # print(train_y)
    return


if __name__ == '__main__':
    main()
