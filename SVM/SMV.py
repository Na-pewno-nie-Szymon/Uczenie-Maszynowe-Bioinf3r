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

def data_loader(PATH: str) -> pd.DataFrame:
    data = pd.read_csv(PATH)
    return data

def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data

def pairplots(data: pd.DataFrame):
    sns.pairplot(data=data, hue='species')
    plt.show()

if __name__=='__main__':
    FILE_PATH = './data/penguins_size.csv'

    # data := [species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]
    data = data_preprocessing(data_loader(FILE_PATH))
    pairplots(data=data)


    print(type(data), data.shape)
    print(data)