'''
TODO:
    [x] Zaimportuj zestaw danych
    [x] autoskalowanie
    [x] przeprowadź modelowanie PLS (jedna zmienna ukryta, 3 deskryptory,
        energia HOMO, polaryzowalność i topologiczny obszar powierzchni)
    [x] Wykreślenie ypred od yobs z podziałem na zbiór uczący i walidacyjny (legenda)
    [ ] Obliczenie statystyk: R2, RMSEc, Q2cvloo, RMSEcvloo, Q2ex, RMSEex
    [ ] Zbudować modele MLR i PCR, powtórzyć dla nich kroki 3 i 4, a następnie
        porównać wyniki uzyskane wszystkimi trzema metodami
    [ ] ypred od yobs dla MLR
    [ ] ypred od yobs dla PCR
    [ ] statystyki dla MLR
    [ ] statyskyki dla PCR
    [ ] porównanie wyników PCR, MLR, PLS
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, DistanceMetric



def data_loader(path: str):
    Yt = pd.read_excel(path, sheet_name="Yt")
    Yv = pd.read_excel(path, sheet_name="Yv")
    Xt = pd.read_excel(path, sheet_name="Xt")
    Xv = pd.read_excel(path, sheet_name="Xv")

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
    yv_pred = pls.predict(X_valid)
    yt_pred = pls.predict(X_train)

    return pls, yv_pred, yt_pred

def y_true_pred(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        set: str
) -> None:
    '''
    ### Functiond that visualise accuracy of prediction model
    @param y_true Validation vector
    @param y_pred Predicted vector
    @param save_fig Decides if figure should be saved if current directory
    '''
    print(y_true.ravel())

    coef = np.polyfit(y_true.ravel(), y_pred.ravel(), 1)
    poly1d_fn = np.poly1d(coef) 
    
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot(y_true, poly1d_fn(y_true), linestyle='--', color='red')
    plt.xlabel('True y values')
    plt.ylabel('Predicted normalised y values')
    plt.grid(visible=True, axis='both')
    plt.savefig(f'./figures/PLS/{set}_ypred_yobs.png')
    plt.show()
    return None

def stats(yt_pred, yv_pred, yt, yv):
    '''
        Obliczenie statystyk: R^2, RMSEc Q^2 CVloo RMSE CVloo, Q^2 Ex RMSE Ex
    '''
    r2 = r2_score(yt, yt_pred)

    RMSEc = np.sqrt(mean_squared_error(yt, yt_pred))

    y_pred_loo = np.zeros_like(yt)  # miejsce na przewidywania LOO

    # Walidacja Leave-One-Out (LOO)
    loo = LeaveOneOut()
    y_true = yt
    for train_index, test_index in loo.split(y_true):
        # Rozdziel dane na treningowe i testowe
        y_train, y_test = y_true[train_index], y_true[test_index]
        # Prosty model: przewidujemy średnią z danych treningowych
        y_pred_loo[test_index] = y_train.mean()

    # RMSE_CVloo
    rmse_cvloo = np.sqrt(mean_squared_error(y_true, y_pred_loo))
    print(f"RMSE_CVloo: {rmse_cvloo:.4f}")

    # Q2_CVloo
    q2_cvloo = 1 - np.sum((y_true - y_pred_loo)**2) / np.sum((y_true - y_true.mean())**2)
    print(f"Q2_CVloo: {q2_cvloo:.4f}")

    q2ex_test = r2_score(yv, yv_pred)

    RMSEex_test = np.sqrt(mean_squared_error(yv, yv_pred))

    print(f'R2: {r2}')
    print(f'RMSEc: {RMSEc}')
    print(f'RMSE_CVloo: {rmse_cvloo}')
    print(f'Q2_CVloo: {q2_cvloo}')
    print(f'Q2_Ex: {q2ex_test}')
    print(f'RMSE_Ex: {RMSEex_test}')

    return 0

def MLR():
    return 0

def PRC():
    return 0

def porównywarka():
    return 0

if __name__ == '__main__':
    FILE_PATH = './data/AA-AuUP.xlsx'

    yt, yv, Xt, Xv = autoscaler(data_loader(FILE_PATH))
    pls, yv_pred, yt_pred = model_PLS(Xt, Xv, yt, yv)

    y_true_pred(yv, yv_pred, 'test')
    y_true_pred(yt, yt_pred, 'validation')
