import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

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


def f(X):
    if (int(X[1]) == 0):
        return 0
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=0.01, algorithm='SAMME.R', random_state=np.abs(int(X[0])), base_estimator=DecisionTreeClassifier(max_depth=np.abs(int(X[1]))))
    ada.fit(train_X, train_Y)
    predict_Y = ada.predict(test_X)
    
    return -f1_score(test_Y, predict_Y, average='macro')


class Particle:
    def __init__(self, lb, ub, dim):
        self.pozicija = np.random.rand(2)
        self.pozicija[0] = np.random.uniform(1, 10000)
        self.pozicija[1] = np.random.uniform(0, 20)
        self.brzina = np.full(dim, 0, dtype=np.dtype("Float64"))
        self.y = np.Inf
        self.najbolji_X = self.pozicija
        self.najbolji_y = self.y


def particle_swarm_optimisation(funkcija, lowerb, upperb, maxiter=20, npart=50, dim=60, tol=10**-15, printData=False):
    w = 1
    dmp = 0.99
    c1 = 2.5
    c2 = 0.5
    # Inicijalizacija
    particles = []
    globalno_najbolji_X = "Neka najbolji pobedi"
    globalno_najbolji_y = np.Inf
    for k in range(npart):
        p = Particle(lowerb, upperb, dim)
        p.y = funkcija(p.pozicija)
        particles.append(p)
        if p.y < globalno_najbolji_y:
            globalno_najbolji_X = p.pozicija
            globalno_najbolji_y = p.y

    # glavna petlja

    for i in range(maxiter):
        for j in range(len(particles)):
            particles[j].brzina = w * particles[j].brzina + c1 * np.random.uniform(size=dim) * (
                    particles[j].najbolji_X - particles[j].pozicija) + c2 * np.random.uniform(size=dim) * (
                                          globalno_najbolji_X - particles[j].pozicija)

            particles[j].pozicija = particles[j].pozicija + particles[j].brzina

            particles[j].y = funkcija(particles[j].pozicija)

            if particles[j].y < particles[j].najbolji_y:
                particles[j].najbolji_X = particles[j].pozicija
                particles[j].najbolji_y = particles[j].y
                if particles[j].y < globalno_najbolji_y:
                    globalno_najbolji_X = particles[j].pozicija
                    if abs(globalno_najbolji_y - particles[j].y) < tol:
                        print("Izlaz! Razbijen kriterijum tolerancije")
                        print(str(i) + ". Globalno Y: " + str(globalno_najbolji_y) + " X: " + str(globalno_najbolji_X))
                        return globalno_najbolji_X
                    globalno_najbolji_y = particles[j].y
            w = w * dmp
            c1 = c1 - 2 / (maxiter*(maxiter-i))
            c2 = c2 + 2 / (maxiter*(maxiter-i))
        if printData:
            print(str(i) + ". Globalno Y: " + str(globalno_najbolji_y) + " X: " + str(np.abs(int(globalno_najbolji_X[0]))) + " " + str(np.abs(int(globalno_najbolji_X[1]))))
    return globalno_najbolji_X


train_file_path = "C:\\Users\\Dusan\\Documents\\github repos\\venice_and_genoa\\src\\zadatak6\\res\\train.csv"
test_file_path = "C:\\Users\\Dusan\\Documents\\github repos\\venice_and_genoa\\src\\zadatak6\\res\\test.csv"
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


particle_swarm_optimisation(f, 0, 1, maxiter=100, npart=50, dim=1, tol=10**-15, printData=True)