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


def f(X):
    if X[0] < 0:
        return 0
    mix = GaussianMixture(n_components=4, random_state=int(X[0]), covariance_type="tied", init_params="random")

    mix.fit(train_X)
    predict_Y = mix.predict(test_X)
    
    return -v_measure_score(test_Y, predict_Y)


class Particle:
    def __init__(self, lb, ub, dim):
        self.pozicija = np.random.rand(1)
        self.pozicija[0] = np.random.uniform(1, 100000)
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
            print(str(i) + ". Globalno Y: " + str(globalno_najbolji_y) + " X: " + str(globalno_najbolji_X))
    return globalno_najbolji_X


train_file_path = "C:\\Users\\Dusan\\Documents\\github repos\\venice_and_genoa\\src\\zadatak5\\res\\train.csv"
test_file_path = "C:\\Users\\Dusan\\Documents\\github repos\\venice_and_genoa\\src\\zadatak5\\res\\test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
train_data = pre_process(train_data)
test_data = pre_process(test_data)
train_data.dropna(inplace=True)
train_data.query("income != 3010", inplace=True) # outliers in africa
train_data.query("income != 1000", inplace=True) # outliers in africa
train_data.query("income != 5523", inplace=True) # outliers in america
train_data.query("income != 4751", inplace=True) # outliers in america
train_data.query("income != 2526", inplace=True) # outliers in asia
train_data.query("income != 1530", inplace=True) # outliers in asia
train_data.query("infant != 43.3", inplace=True) # outliers in europe
print(train_data)
train_X = train_data.iloc[:, [0, 1, 3]]
train_Y = train_data.iloc[:, 2]
test_X = test_data.iloc[:, [0, 1, 3]]
test_Y = test_data.iloc[:, 2]


particle_swarm_optimisation(f, 0, 1, maxiter=100, npart=2000, dim=1, tol=10**-15, printData=True)