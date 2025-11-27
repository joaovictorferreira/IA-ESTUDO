import numpy as np
import math
import matplotlib.pyplot as plt

class NaiveBayes():
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

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

    X = np.column_stack((income, age))
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

    plt.pie(
        [acertos, erros],
        labels=[f"Acertos ({acertos})", f"Erros ({erros})"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0),
        shadow=True
    )

    plt.tight_layout()
    plt.show()

X, y = processarDados()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

naive_b = NaiveBayes()
naive_b.fit(X_train, y_train)
pred = naive_b.predict(X_test)

gerarGraficosPerceptron(pred, y_test, ["Income", "Age"])