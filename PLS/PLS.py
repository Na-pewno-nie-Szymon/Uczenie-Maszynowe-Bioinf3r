
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

def data_loader(path: str):
    # Wczytanie danych z pliku Excel
    Yt = pd.read_excel(path, sheet_name="Yt")
    Yv = pd.read_excel(path, sheet_name="Yv")
    Xt = pd.read_excel(path, sheet_name="Xt")
    Xv = pd.read_excel(path, sheet_name="Xv")
    return [Yt, Yv, Xt, Xv]

def autoscaler(matrix):
    # Usunięcie pierwszej kolumny (zakładamy, że zawiera indeksy lub nazwy)
    matrix.drop(matrix.columns[0], axis=1, inplace=True)
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)

def model_PLS(X_train, X_valid, Y_train, Y_valid, n_components=1):
    # Tworzenie i trenowanie modelu PLS
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, Y_train)
    return pls, pls.predict(X_valid), pls.predict(X_train)

def model_MLR(X_train, X_valid, Y_train, Y_valid):
    # Tworzenie i trenowanie modelu regresji liniowej
    mlr = LinearRegression()
    mlr.fit(X_train, Y_train)
    return mlr, mlr.predict(X_valid), mlr.predict(X_train)

def model_PCR(X_train, X_valid, Y_train, Y_valid, n_components=3):
    # Redukcja wymiarowości za pomocą PCA, a następnie regresja liniowa
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    mlr = LinearRegression()
    mlr.fit(X_train_pca, Y_train)
    return mlr, mlr.predict(X_valid_pca), mlr.predict(X_train_pca)

def y_true_pred(y_true, y_pred, set_name):
    # Wizualizacja predykcji względem rzeczywistych wartości
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, color='blue', label=f'{set_name} Data')
    coef = np.polyfit(y_true.ravel(), y_pred.ravel(), 1)  # Dopasowanie linii regresji
    plt.plot(y_true, np.poly1d(coef)(y_true), linestyle='--', color='red', label='Fit Line')
    plt.xlabel('True y values')
    plt.ylabel('Predicted y values')
    plt.legend()
    plt.grid()
    plt.savefig(f'F:\\vs_code_workspace\\uczenie_maszynowe\\Uczenie-Maszynowe-Bioinf3r\\PLS\\{set_name}_ypred_yobs.png')
    plt.show()

def stats(y_true, y_pred, name):
    # Obliczenie miar dopasowania modelu
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    q2_ex = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    # Walidacja Leave-One-Out (LOO)
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y_true)
    for train_index, test_index in loo.split(y_true):
        y_train, y_test = y_true[train_index], y_true[test_index]
        y_pred_loo[test_index] = np.mean(y_train)  # Przewidywanie średnią wartością
    
    rmse_cvloo = np.sqrt(mean_squared_error(y_true, y_pred_loo))
    q2_cvloo = 1 - (np.sum((y_true - y_pred_loo)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    print(f'{name} R2: {r2:.4f}, RMSE: {rmse:.4f}, Q2_Ex: {q2_ex:.4f}, RMSE_CVloo: {rmse_cvloo:.4f}, Q2_CVloo: {q2_cvloo:.4f}')
    return r2, rmse, q2_ex, rmse_cvloo, q2_cvloo

def compare_models(models):
    # Porównanie wyników uzyskanych różnymi modelami
    for name, values in models.items():
        print(f'Model: {name}, R2: {values[0]:.4f}, RMSE: {values[1]:.4f}, Q2_Ex: {values[2]:.4f}, RMSE_CVloo: {values[3]:.4f}, Q2_CVloo: {values[4]:.4f}')

if __name__ == '__main__':
    FILE_PATH = 'F:\\vs_code_workspace\\uczenie_maszynowe\\Uczenie-Maszynowe-Bioinf3r\\data\\AA-AnUP.xlsx'
    yt, yv, Xt, Xv = data_loader(FILE_PATH)
    yt, yv, Xt, Xv = map(autoscaler, [yt, yv, Xt, Xv])
    
    models = {}
    for model_name, model_func in zip(["PLS", "MLR", "PCR"], [model_PLS, model_MLR, model_PCR]):
        model, yv_pred, yt_pred = model_func(Xt, Xv, yt, yv)
        y_true_pred(yv, yv_pred, model_name + '_Test')
        y_true_pred(yt, yt_pred, model_name + '_Train')
        models[model_name] = stats(yv, yv_pred, model_name)
    
    compare_models(models)
