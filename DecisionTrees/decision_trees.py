'''
    Zadanie 1. (3 punkty)
    Dane wejściowe – ftalany_klasyfikacja.xlsx zawiera dane dotyczące 32
    ftalanów. Dane są już po autoskalowaniu. Podział na zbiór treningowy i walidacyjny znajdują
    się w poszczególnych arkuszach. Związki są podzielone na dwie kategorie: 1 (trwałość w
    przedziale dni-tygodnie) oraz 2 (tygodnie-miesiące).

    1. Przygotuj macierz korelacji pomiędzy zmiennymi niezależnymi i zmienną zależną.
    2. Sprawdź czy zbiór testowy i treningowy są zbalansowane.
    3. Zbuduj model drzewa klasyfikacyjnego w celu przewidywania tego parametru
    (kategorii trwałości). Oceń zdolności prognostyczne modelu na podstawie macierzy
    pomyłek oraz statystyk: czułości, specyficzności, precyzji, współczynnika F1,
    dokładności oraz zbalansowanego błędu.
    4. Dokonaj optymalizacji parametrów dwiema metodami (z uzasadnieniem ich wyboru)


    Zadanie 2. (2 punkty)
    Dane wejściowe – ftalany.xlsx zawiera dane dotyczące 32 ftalanów. Dane są
    już po autoskalowaniu. Podział na zbiór treningowy i walidacyjny znajdują się w
    poszczególnych arkuszach.

    1. Zbuduj model drzewa regresyjnego, aby przewidzieć stałą szybkości degradacji
    poszczególnych związków.
    2. Oblicz statystyki R2, RMSE, Q2 i RMSEex. Narysuj wykres zależności ypred od yobs,
    oraz wykres słupkowy zależności ypred od yobs.
    3. Dokonaj optymalizacji parametrów dwiema metodami (z uzasadnieniem ich wyboru)
    4. Dokonaj interpretacji uzyskanych wyników
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import balanced_accuracy_score

def zadanie1():
    X_train = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'X_train')
    y_train = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'y_train')
    X_test = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'X_test')
    y_test = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'y_test')

    # 1. Przygotuj macierz korelacji pomiędzy zmiennymi niezależnymi i zmienną zależną.
    Xy_matrix = pd.concat([X_train, y_train], axis=1)
    corr_matrix = correlationMatrix(Xy_matrix)
    print('Correlation matrix:')
    print(corr_matrix, '\n')

    # 2. Sprawdź czy zbiór testowy i treningowy są zbalansowane.
    check_balance(y_train, 'treningowy')
    print()
    check_balance(y_test, 'testowy')
    print()

    # 3. Zbuduj model drzewa klasyfikacyjnego w celu przewidywania tego parametru
    decisionTreeClassifier(X_train, y_train, X_test, y_test)
    
    # 4. Dokonaj optymalizacji parametrów dwiema metodami (z uzasadnieniem ich wyboru)
    decisionTreeClassifierOptimization(X_train, y_train, X_test, y_test)


def dataLoader(path: str, sheet_name: str) -> pd.DataFrame:
    '''
    ### Funkcja do wczytywania danych z pliku Excel z określonego arkusza ###
    * path: ścieżka do pliku Excel
    * sheet_name: nazwa arkusza
    * return: DataFrame z danymi z arkusza
    '''
    data = pd.read_excel(path, sheet_name=sheet_name)
    return data

def correlationMatrix(data: pd.DataFrame, visualise: bool = True) -> pd.DataFrame:
    '''
    ### Funkcja do tworzenia i wizualizacji macierzy korelacji ###
    * data: DataFrame z danymi
    * visualise: flaga określająca czy wyświetlić wykres heatmapy
    * return: DataFrame z macierzą korelacji
    '''
    correlation_matrix = data.corr()

    if visualise:
        # Tworzenie wykresu heatmapy dla macierzy korelacji
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Macierz Korelacji')
        plt.savefig('./DecisionTrees/macierz_korelacji.png')
        plt.show()
    

    return correlation_matrix

def check_balance(y: pd.Series, set_name: str) -> None:
    '''
    ### Funkcja sprawdzająca czy zbiór jest zbalansowany ###
    * y: etykiety zbioru
    * set_name: nazwa zbioru
    '''
    counts = y.value_counts(normalize=True)
    print(f"Balans dla {set_name}:")
    print(counts)
    if counts.min() < 0.4:  # Przykładowy próg
        print(f"Uwaga: Zbiór {set_name} jest niezbalansowany!")
    else:
        print(f"Zbiór {set_name} jest zbalansowany.")

def decisionTreeClassifier(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    '''
    ### Funkcja do budowy modelu drzewa klasyfikacyjnego i oceny jego zdolności prognostycznych ###
    * X_train: zbiór treningowy zmiennych niezależnych
    * y_train: zbiór treningowy etykiet
    * X_test: zbiór testowy zmiennych niezależnych
    * y_test: zbiór testowy etykiet
    '''
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Macierz pomyłek:")
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    specificity = recall_score(y_test, y_pred, average='binary', pos_label=2)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
    balanced_error = 1 - balanced_accuracy_score(y_test, y_pred)

    print("Statystyki modelu:")
    print(f"Dokładność: \t\t{accuracy:.2f}")
    print(f"Czułość (Recall): \t{recall:.2f}")
    print(f"Specyficzność: \t\t{specificity:.2f}")
    print(f"Precyzja: \t\t{precision:.2f}")
    print(f"Współczynnik F1: \t{f1:.2f}")
    print(f"Zbalansowany błąd: \t{balanced_error:.2f}")

def decisionTreeClassifierOptimization(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    pass


if __name__ == '__main__':
    zadanie1()
