'''
1.  Zaimportowanie autoskalowanego zestawu danych.
    Arkusze Yt oraz Yv zawierają wartości modelowanej wielkości
    (energia absorbcji aminokwasu na powierzchni nanocząstki złota)
    odpowiednio dla związków ze zbioru uczącego i awlidacyjnego;
    arkusze Xt i Xv zawierają deskryptory obliczone odpowiednio dla
    związków zbioru uczącego i walidacyjnego

2.  Przeprowadzenie modelowania PLS (jedna ukryta zmienna; 3 deskryptory:
    energia HOMO, polaryzowalność i topologiczny obszar powierzchni)

3.  Wykreślenie ypred od yobs z podziałem na zbiór uczący i walidacyjny (legenda)

4.  Obliczenie statystyk: R^2, RMSEc Q^2 CVloo RMSE CVloo, Q^2 Ex RMSE Ex

5.  Zbudować modele MLR i PRC, powtórzyć dla nich kroki 3 i 4, a następnie
    porównać wyniki uzyskane wszystkimi trzema metodami.
'''
import pandas as pd
import os
from sys import argv
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

def data_loader(path: str):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Plik {path} does not exist!')
        
        Yt = pd.read_excel(path, sheet_name="Yt")
        Yv = pd.read_excel(path, sheet_name="Yv")
        Xt = pd.read_excel(path, sheet_name="Xt")
        Xv = pd.read_excel(path, sheet_name="Xv")

    except FileNotFoundError as error:
        print(f'Error: {error}')
        exit()
    except Exception as error:
        print(f"Unknown error ocured: {error}")
        exit()

    return [Yt, Yv, Xt, Xv]

def autoscaler(matrix):
    '''
        Funkcja do autoskalowania danych
        mean    =   0
        std_dev =   1
    '''
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    return scaled_matrix

def model_PLS(X_train, X_valid, Y_train, Y_valid, n_components = 1):
    '''
        Funkcja trenująca model PLS i oceniająca na danych validacyjnych
    '''
    pls = PLSRegression(n_components=n_components)

    # Trening modelu
    pls.fit(X_train, Y_train)

    # Predykcja na zbiorze testowym
    Y_pred = pls.predict(X_valid)

    # Obliczenie MSE (mean squared error)
    mse = mean_squared_error(Y_valid, Y_pred)

    return pls, mse, Y_pred

def yp_yo_plot():
    return 0

def stats():
    '''
        Obliczenie statystyk: R^2, RMSEc Q^2 CVloo RMSE CVloo, Q^2 Ex RMSE Ex
    '''
    return 0

def MLR():
    return 0

def PRC():
    return 0

def porównywarka():
    return 0

if __name__ == '__main__':
    try:
        if argv[1] == '-h' or argv[1] == 'help':
            print('use: "python3 PLS.py file_path.xlsx"')
            print('Avaliable flags:')
            print('-h       :      help message')
            print('-help    :      help message')
            print('-f       :      file path')
            print('-n_comp <n>:    <n> latent varible(s) when not used n = 1')

        else:
            # Wczytywanie danych
            DATA_PATH = argv[1]
            Yt, Yv, Xt, Xv = data_loader(DATA_PATH)
            if Yt is not None:
                print('Data loaded succesfully')

            # Autoskalowanie danych
            Yt = autoscaler(Yt)
            Yv = autoscaler(Yv)
            Xt = autoscaler(Xt)
            Xv = autoscaler(Xv)
            if Yt == autoscaler(Yt):
                print('Data autoscaled succesfully')

            # Modelowanie PLS z jedną ukrytą zmienną
            pls_model, mse, Y_pred = model_PLS(Xt, Xv, Yt, Yv)

            print('PLS model generated succesfully')
            print(f'Mean Squared Error: {mse}')

    except Exception as error:
        print(type(error).__name__)
        print('Something went wrong!')
        print('Run with -h/-help flag to get more information')

