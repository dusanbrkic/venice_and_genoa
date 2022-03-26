import sys
import numpy as np
import pandas
# from matplotlib import pyplot as plot

OUTLIER_BIAS_DIFFERENCE = 25


class Ridge:
    def __init__(self, alfa=1):
        self.alfa = alfa
        self.koeficijenti = None

    def treniraj(self, X, y):
        dim = X.shape
        jedinicna = np.identity(dim[1])

        self.koeficijenti = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + self.alfa * jedinicna), X.T), y)
        return

    def predvidi(self, parametri):
        y = [0 for o in range(len(parametri))]
        for j in range(len(parametri)):
            y[j] = self.koeficijenti[0]
            for i in range(1, len(self.koeficijenti)):
                y[j] = y[j] + parametri[j][i] * self.koeficijenti[i]
        return y


class Gradijent:
    def __init__(self, ucenje=0.01, iteracije=10000):
        self.koeficijenti = None
        self.ucenje = ucenje
        self.iteracije = iteracije

    def _eval(self, theta, X, y):

        n = y.shape[0]
        predvidjanja = X.dot(theta)
        vrednost = (1 / 2 * n) * np.sum(np.square(predvidjanja - y))
        return vrednost

    def treniraj(self, X, y):
        n = len(y)
        theta = np.random.rand(X.shape[1], 1)

        for it in range(self.iteracije):
            vrednost = 0.0
            for i in range(n):
                random = np.random.randint(0, n)
                X_i = X[random, :].reshape(1, X.shape[1])
                y_i = y[random].reshape(1, 1)
                predvidjanje = np.dot(X_i, theta)
                theta = theta - (1 / n) * self.ucenje * (X_i.T.dot((predvidjanje - y_i)))
                vrednost += self._eval(theta, X_i, y_i)
        self.koeficijenti = theta

    def predvidi(self, parametri):
        y = [0 for o in range(len(parametri))]
        print("koef:" + str(self.koeficijenti))
        for j in range(len(parametri)):
            y[j] = self.koeficijenti[0][0]
            for i in range(1, len(self.koeficijenti)):
                y[j] = y[j] + parametri[j][i] * self.koeficijenti[i][0]
        return y


def dodaj_jedinice(X):
    dim = X.shape
    jedinice = np.full((dim[0], 1), 1)
    X = np.c_[jedinice, X]
    return X


def stepenuj_matricu(X, stepen):
    Y = X
    for i in range(1, stepen):
        X = np.c_[X, Y ** (i + 1)]
    return X


def rmse(true_y, pred_y):
    return np.sqrt(np.mean((pred_y - true_y) ** 2))


def get_outliers_idxs(pred_Y, train_y):
    ret = []
    for i in range(len(pred_Y)):
        if np.abs(pred_Y[i] - train_y[i]) > OUTLIER_BIAS_DIFFERENCE:
            ret.append(i)
    return ret

def main():
    # read data
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    train_data = pandas.read_csv(train_file_path)
    test_data = pandas.read_csv(test_file_path)
    train_X = train_data.iloc[:, 0]
    train_y = train_data.iloc[:, 1]
    test_X = test_data.iloc[:, 0]
    test_y = test_data.iloc[:, 1]

    # train the model
    regression = Ridge(0.0000001)
    train_X = stepenuj_matricu(train_X, 4)
    train_X = dodaj_jedinice(train_X)
    regression.treniraj(train_X, train_y)

    # remove the outliers
    pred_Y = regression.predvidi(train_X)
    outliers = get_outliers_idxs(pred_Y, train_y)
    train_X_proc = np.delete(train_X, outliers, axis=0)
    train_y_proc = np.delete(train_y.to_numpy(), outliers, axis=0)

    # train the model again
    regression.treniraj(train_X_proc, train_y_proc)

    # test the model
    test_X = stepenuj_matricu(test_X, 4)
    test_X = dodaj_jedinice(test_X)
    pred_Y = regression.predvidi(test_X)

    # plot.scatter(train_X, train_y, color='blue')
    #
    # plot.scatter(train_X, rez2, color='red')
    # plot.show()

    print(rmse(test_y, pred_Y))

    return rmse(test_y, pred_Y)


if __name__ == '__main__':
    main()
