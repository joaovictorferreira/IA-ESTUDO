import numpy as np
import math
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None 
        self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._step_function(linear_output)
                if y[idx] != y_predicted:
                    update = self.learning_rate * (y[idx] - y_predicted)
                    self.weights += update* x_i 
                    self.bias += update
    def predict(self, X):
        linear_output = np.dot(X, self.weights) +self.bias
        y_predicted = self._step_function(linear_output)
        return y_predicted
    def _step_function(self, x):
        return np.where(x>=0, 1, 0)
    
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

    X = np.column_stack((age, income))
    y = np.array([1 if v == "Yes" else 0 for v in purchased_bike])
    return X, y

def acuracia_matriz(predicoes, reais):
    acertos = sum(p == r for p, r in zip(predicoes, reais))
    erros = len(predicoes) - acertos
    total = len(predicoes)
    acuracia = (acertos / total) * 100
    return acuracia, acertos, erros, total

def f1_score_matriz(predicoes, reais):
    TP = sum((predicoes == 1) & (reais == 1))
    FP = sum((predicoes == 1) & (reais == 0))
    FN = sum((predicoes == 0) & (reais == 1))

    if TP == 0 and (FP == 0 or FN == 0):
        return 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def gerarGraficosPerceptron(predicoes, reais, features_usadas):
    acuracia, acertos, erros, total = acuracia_matriz(predicoes,reais)
    f1_score = f1_score_matriz(predicoes, reais)
    lista_features = ", ".join(features_usadas)

    titulo  = f"Acurácia: {acuracia:.2f}%\n"
    titulo += f"F1-Score: {f1_score:.2f}%\n"
    titulo += f"Total de amostras: {total}\n"
    titulo += f"Atributos utilizados: {lista_features}"

    plt.figure(figsize=(6,6))
    plt.title(titulo, fontsize=11)

    plt.pie([acertos, erros],
            labels=[f"Acertos ({acertos})", f"Erros ({erros})"],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.05, 0),
            shadow=True)

    plt.tight_layout()
    plt.show()

X,y = processarDados();
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

p = Perceptron(learning_rate=0.1, epochs=500)
p.fit(X_train, y_train)
predicoes = p.predict(X_test)
gerarGraficosPerceptron(predicoes, y_test, ["Age", "Income"])