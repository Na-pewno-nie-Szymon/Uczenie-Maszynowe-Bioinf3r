'''
    Stwórz model klasyfikacyjny metodą wektorów nośnych wg poniższych wytycznych:
    [x] 1 Wczytaj zbiór danych "penguins_size.csv" zawierający 
        charakterystykę trzech gatunków pingwinów 
    [x] 2 Wytnij brakujące wartości
    [x] 3 Stwórz wykresy pokazujące relację pomiędzy zmiennymi (seaborn, pairplot)
    [ ] 4 Na podstawie wykresów wybierz:
        - dwie cechy oraz dwa gatunki (dwie klasy), dla których
          obiekty są liniowo spreparowane
        - dwie cechy oraz dwa gatunki (dwie klasy), dla których
          obiekty nie są liniowo sprwparowane
    [ ] 5 Dla obiektów liniowo spreparowanych zbuduj model z wykorzystaniem
        liniowej funkcji jądra (kernel='linear') i policz dokładność, z jaką
        przewiduje. Liczebność zbioru walidacyjnego powinna wynosić 20% całego
        zbioru danych. Pamiętaj o autoskalowaniu danych. Obiekty w przestrzeni
        cech przedstaw na wykresie z zaznaczeniem wektorów własnych oraz marginesem.
    [ ] 6 Dla obiektów liniowo niesparowanych zbuduj modele z wykorzystaniem:
        - liniowej funcki jądra (kernel='linear')
        - funkcji wielomianowej jądra (kernel='poly')
        - radialnej funkcji bazowej (kerbel='rbf')
        Dla każdego z nich oblicz dokładność. Liczebność zbioru walidacyjnego
        we wszystkich przypadkach powinna wynosić 20% całego zbioru danych.
        Pamiętaj o autoskalowaniu danych. Obiekty w przestrzeni cech przedstaw
        na wykresie (dla kernel='linear') z zaznaczeniem wektorów własnych oraz
        marginesem
    [ ] 7 Krótko skomentuj uzyskane wyniki
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

def data_loader(PATH: str) -> pd.DataFrame:
    data = pd.read_csv(PATH)
    return data

def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data

def pairplots(data: pd.DataFrame):
    sns.pairplot(data=data, hue='species')
    plt.show()

def plot_svm_decision_boundary(model, X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    sv = model.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.legend()
    plt.title(title)
    plt.show()

if __name__=='__main__':
    FILE_PATH = './data/penguins_size.csv'

    data = data_preprocessing(data_loader(FILE_PATH))
    pairplots(data=data)

    # Wybór dwóch cech i dwóch gatunków do analizy liniowej
    selected_species = data[data['species'].isin(['Adelie', 'Chinstrap'])]
    X = selected_species[['bill_length_mm', 'bill_depth_mm']].values
    y = selected_species['species'].apply(lambda x: 1 if x == 'Adelie' else 0).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model liniowy
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred = svm_linear.predict(X_test)
    accuracy_linear = accuracy_score(y_test, y_pred)
    print(f'Linear Kernel Accuracy: {accuracy_linear:.2f}')
    plot_svm_decision_boundary(svm_linear, X_train, y_train, "Linear Kernel")
    
    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
            svm_model = SVC(kernel=kernel)
            svm_model.fit(X_train, y_train)

            y_pred = svm_model.predict(X_test)

            print(f'Accuracy ({kernel}, non-separable): {accuracy_score(y_test, y_pred)}')
    
    # Wizualizacja dla jądra liniowego (nieliniowo separowalne dane)
    plt.figure()

    plot_decision_regions(X_train, y_train, clf=SVC(kernel='linear').fit(X_train, y_train))

    plt.title("Decision Boundary for Non-linear SVM (Linear Kernel)")

    plt.xlabel("B Length (scaled)")

    plt.ylabel("Body Mass (scaled)")

    plt.show()