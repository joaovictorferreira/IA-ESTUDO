import numpy as np
import math
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=5, task='classification'):
        self.k = k
        self.task = task

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def calculate_prediction(self, x):
        distances = [self.euclidian_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        if self.task == "classification":
            unique, counts = np.unique(k_labels, return_counts=True)
            return unique[np.argmax(counts)]

        elif self.task == "regression":
            return np.mean(k_labels)

        else:
            raise ValueError("Tarefa não definida")
   
def processarDados():
    dados = np.loadtxt("C:\\Users\\João\Documents\\faculdade\\IA\\AV3\\bike_buyers.csv", delimiter=",", skiprows=1, dtype=str)

    marital_status = dados[:,1]
    gender = dados[:,2]
    income = dados[:,3]
    children = dados[:,4]
    education = dados[:,5]
    occupation = dados[:,6]
    home_owner = dados[:,7]
    cars = dados[:,8]
    commute_distance = dados[:,9]
    region = dados[:,10]
    age = dados[:,11]
    purchased_bike = dados[:,12]

    filtro = (age != "") & (income != "") & (purchased_bike != "") & (children != "")

    income = income[filtro]
    age = age[filtro]
    children = children[filtro]
    purchased_bike = purchased_bike[filtro]

    age = age.astype(float)
    income = income.astype(float)
    children = children.astype(float)

    X = np.column_stack((age, income, children))
    y = purchased_bike
    return X, y

def gerarGraficos(predicoes, reais, k, features_usadas):
    acertos = sum(p == r for p, r in zip(predicoes, reais))
    erros = len(predicoes) - acertos
    total = len(predicoes)
    acuracia = (acertos / total) * 100
    lista_features = ", ".join(features_usadas)

    titulo1  = f"Acurácia: {acuracia:.2f}%\n"
    titulo1 += f"Total de amostras: {total}\n"
    titulo1 += f"Número de K utilizado: {k}\n"
    titulo1 += f"Atributos utilizados: {lista_features}"

    plt.figure(figsize=(6,6))
    plt.title(titulo1, fontsize=11)

    plt.pie([acertos, erros],
            labels=[f"Acertos ({acertos})", f"Erros ({erros})"],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.05, 0),
            shadow=True)

    plt.tight_layout()
    plt.show()

def train_test_split(X, y, test_size=0.2, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)

    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_test = math.ceil(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices, :], X[test_indices, :]

    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

X, y = processarDados()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn = KNN(k, task="classification")
knn.fit(X_train, y_train)
predicoes = [knn.calculate_prediction(x) for x in X_test]

gerarGraficos(predicoes, y_test, k, ["Age", "Income", "Children"])