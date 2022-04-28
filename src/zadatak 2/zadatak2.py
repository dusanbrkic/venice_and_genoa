import sys
import numpy as np
import pandas


class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1, length: float = 1):
        self.sigma_f = sigma_f
        self.length = length

    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        return float(self.sigma_f *
                     np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) /
                            (2 * self.length**2)))


class KRRS:
    def __init__(self, trainData, kernelFunc, lambdaPara):
        self.trainX = trainData[0]
        trainY = trainData[1]

        self.kernelFunc = kernelFunc

        kArr = np.empty(
            (self.trainX.shape[0], self.trainX.shape[0]), dtype=np.float64)  # zeros

        for i in range(0, self.trainX.shape[0]):
            for j in range(0, self.trainX.shape[0]):
                xi = self.trainX[i]
                xj = self.trainX[j]

                kij = kernelFunc(xi, xj)

                kArr[i][j] = kij

        ridgeParas = lambdaPara * \
            np.identity(self.trainX.shape[0], dtype=np.float64)

        # alpha for kernel ridge $\alpha = (\Phi(X)\phi^T(X)+\lambda I)^{-1}Y$
        self.alpha = np.dot(np.linalg.inv(np.add(kArr, ridgeParas)), trainY)

    def predict(self, testX):
        YPred = np.empty((testX.shape[0]), dtype=np.float64)  # zeros
        for testInd in range(0, testX.shape[0]):

            xnew = testX[testInd]
            # for i in range(0, trainX.shape[0]):   # $y_{new} = \sum_{i}  \alpha_i \Phi(x_i) \Phi(x_{new})
            # innerVal =
            YPred[testInd] = np.sum([np.dot(self.alpha[i], self.kernelFunc(
                self.trainX[i], xnew)) for i in range(0, self.trainX.shape[0])])  # sum ??

        return YPred


class Ridge:
    def __init__(self, alfa=1):
        self.alfa = alfa
        self.koeficijenti = None

    def fit(self, X, y):
        dim = X.shape
        jedinicna = np.identity(dim[1])

        self.koeficijenti = np.dot(np.dot(np.linalg.inv(
            np.dot(X.T, X) + self.alfa * jedinicna), X.T), y)

        return self.koeficijenti

    def predict(self, parametri):
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
    print(sys.argv)
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    train_data = pandas.read_csv(train_file_path)
    test_data = pandas.read_csv(test_file_path)

    zvanja = pandas.get_dummies(train_data.iloc[:, 0])
    oblasti = pandas.get_dummies(train_data.iloc[:, 1])
    train_data = zvanja.join(oblasti).join(train_data.iloc[:, [2,3,5]])

    zvanja_test = pandas.get_dummies(test_data.iloc[:, 0])
    oblasti_test = pandas.get_dummies(test_data.iloc[:, 1])
    test_data = zvanja_test.join(oblasti_test).join(test_data.iloc[:, [2,3,5]])

    train_X = train_data.iloc[:, 0:-1].to_numpy()
    train_y = train_data.iloc[:, -1].to_numpy()
    test_X = test_data.iloc[:, 0:-1].to_numpy()
    test_y = test_data.iloc[:, -1].to_numpy()

    train_X = stepenuj_matricu(train_X, 1)
    train_X = dodaj_jedinice(train_X)
    test_X = stepenuj_matricu(test_X, 1)
    test_X = dodaj_jedinice(test_X)

    # regression = Ridge(0.001389)
    # regression.fit(train_X, train_y)
    # pred_Y = regression.predict(test_X)

    # print(rmse(test_y, pred_Y))

    model = KRRS([train_X, train_y],
                 SquaredExponentialKernel(5, 10),
                 0.01
                 )
    pred_Y = model.predict(test_X)

    print(rmse(test_y, pred_Y))

    return


if __name__ == '__main__':
    main()
