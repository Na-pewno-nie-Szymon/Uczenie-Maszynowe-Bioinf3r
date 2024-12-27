import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def data_loader(path: str):
    '''
        1. Import standaryzowanego zestawu danych 'dane_leki.xlsx'
    '''
    data = pd.read_excel(path)
    return data

def data_setter(data):
    '''
    Splitting dataset to Y vector and X matrix
    '''
    Y_vector = data[['logK HSA']]
    X_matrix = data[['logKCTAB', 'CATS3D_00_DD', 'CATS3D_09_AL', 'CATS3D_00_AA']]

    return Y_vector, X_matrix

def PCA_(X_matrix):
    '''
        2. Przeprowadzenie analizy PCA (biblioteka: sklearn.decomposition.pca)
    '''
    pca = PCA(n_components=0.8)
    X_pca = pca.fit_transform(X_matrix)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance

def plot_pca_variance(explained_variance):
    '''
    Tworzenie wykresu wariancji wyjaśnionej przez komponenty PCA
    '''
    n_components = len(explained_variance)
    components = np.arange(1, n_components + 1)

    plt.figure(figsize=(8, 5))
    plt.bar(components, explained_variance, alpha=0.6, label='Wariancja wyjaśniona')
    plt.plot(components, np.cumsum(explained_variance), marker='o', color='r', label='Skumulowana wariancja')

    plt.xlabel('Liczba komponentów')
    plt.ylabel('Wariancja wyjaśniona przez kolejne komponenty PCA')
    plt.title('Wariancja wyjaśniona przez kolejne komponenty PCA')
    plt.legend()
    plt.grid(True)
    plt.show()


def dataset_splitter(X_matrix, Y_vector, test_size=0.33, random_state=42):
    '''
        3. Podział na zbiór uczący i walidacyjny (sklearn.model_selection):
        - train_test_split -> test_size = 0.33, random_state = 42
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X_matrix, Y_vector, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

def RMSEc(X_train, Y_train):
    '''
        4. Wykreślenie zależności RMSEc od liczby uwzględnianych głównych składowych:
        - metoda walidacji krzyżowej KFold (sklearn.model_selection) -> n_splits = 10, shuffle = True, random_state = 0
    '''
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    rmse_values = []

    max_components = X_train.shape[1]

    for n_components in range(1, max_components + 1):
        X_subset = X_train.iloc[:, :n_components]
        rmse_fold = []

        for train_index, val_index in kfold.split(X_subset):
            X_train_fold, X_val_fold = X_subset.iloc[train_index], X_subset.iloc[val_index]
            Y_train_fold, Y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]

            model = LinearRegression()
            model.fit(X_train_fold, Y_train_fold)
            Y_pred = model.predict(X_val_fold)

            rmse = np.sqrt(mean_squared_error(Y_val_fold, Y_pred))
            rmse_fold.append(rmse)
        rmse_values.append(np.mean(rmse_fold))
    return rmse_values

def MLR(X_train, X_val, Y_train, Y_val):
    '''
        5. Zbudowanie modelu regresji liniowej dla istotnej liczby głównych składowych
    '''
    pca = PCA()
    pca.fit(X_train)

    optimal_components = 3  # najniższa wartość RMSEc
    X_train_pca = pca.transform(X_train)[:, :optimal_components]
    X_val_pca = pca.transform(X_val)[:, :optimal_components]

    model = LinearRegression()
    model.fit(X_train_pca, Y_train)
    Y_pred = model.predict(X_val_pca)

    rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
    r2 = r2_score(Y_val, Y_pred)

    print(f'RMSE:   {rmse}')
    print(f'R^2:    {r2}')
    pass

def RMSE_plot(rmse_values):
    '''
        Tworzenie wykresu RMSE w zaleznosci od liczby komponentow
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rmse_values) + 1), rmse_values, marker='o')
    plt.title('Wartosci RMSE w zaleznosci od liczby glownych skladowych')
    plt.xlabel('Liczba glownych skladowych')
    plt.ylabel('RMSE')
    plt.grid()
    plt.show()

def stats():
    '''
        6. Obliczenie R^2 i RMSE oddzielnie dla zbiorów kalibracyjnego i walidacyjnego
    ''' 
    pass

if __name__=="__main__":
    '''
        Proszę, krótko zinterpretować wykres i uzyskane wyniki
    '''
    try:
        path = sys.argv[1]
        if path == '-h' or path == '-help':
            print('to use this program copy this line:')
            print('python3 PCR.py [path to data]')
        else:
            data = data_loader(path)
            y, X = data_setter(data)
            X_pca, explained_variation = PCA_(X_matrix=X)
            plot_pca_variance(explained_variation)
            X_train, X_test, Y_train, Y_test = dataset_splitter(X, y)
            rmse_list = RMSEc(X_train, Y_train)
            RMSE_plot(rmse_list)
            MLR(X_train, X_test, Y_train, Y_test)
    except IndexError:
        print('for more information use -h or -help flag')
        print('python3 PCR.py -h        |   python3 PCR.py -help')
    finally:
        print(len(sys.argv))

