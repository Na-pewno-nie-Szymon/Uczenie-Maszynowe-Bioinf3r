'''
1. sprawdzenie korelacji pomiędzy zmiennymi objaśniającymi (corr_matrix)
2. równanie modelu
3. wykreślenie ypred od yobs z podziałem na zbiór uczącu i walidacyjny (legenda)
4. sprawdzenie dziedziny modelu (wykres Williamsa z zaznaczoną wartością graniczną, podział na zbiór uczący i walidacyjny)
5. obliczenie statystyk:
    - R^2
    - RMSE C
    - Q^2 CVloo
    - RMSE CVloo
    - Q^2 Ex
    - RMSE Ex
    - F
  
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

def wykres_Williamsa(X_train, Y_train, X_test, Y_test):
    X_train_with_const = sm.add_constant(X_train)
    ols_model = sm.OLS(Y_train, X_train_with_const).fit()
    influence = ols_model.get_influence()
    leverage_train = influence.hat_matrix_diag
    standardized_residuals_train = influence.resid_studentized_internal

    X_test_with_const = sm.add_constant(X_test)
    ols_model = sm.OLS(Y_test, X_test_with_const).fit()
    influence = ols_model.get_influence()
    leverage_test = influence.hat_matrix_diag
    standardized_residuals_test = influence.resid_studentized_internal

    # Granica dla dźwigni
    n, p = X_train.shape
    h_limit = 3 * (p + 1) / n

    plt.scatter(leverage_train, standardized_residuals_train, label="Zbiór uczący", color='blue')
    plt.scatter(leverage_test, standardized_residuals_test, label="Zbiór walidacyjny", color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axhline(y=3, color='red', linestyle='--', label="Granica reszt")
    plt.axhline(y=-3, color='red', linestyle='--')
    plt.axvline(x=h_limit, color='green', linestyle='--', label=f"Granica dźwigni={h_limit:.3f}")
    plt.xlabel("Dźwignia")
    plt.ylabel("Reszty standaryzowane")
    plt.title("Wykres Williamsa")
    plt.legend()
    plt.show()

def wykres_ypred_yobs(model, X_matrix, Y_vector):# Predykcja na obu zbiorach
    Y_obs_train = Y_vector.iloc[:19]
    Y_obs_test = Y_vector.iloc[19:]
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    Y_pred_train = []
    Y_pred_test = []
    for id in range(len(Y_vector)):
        y = intercept
        i = 0
        for key in X_matrix:
            y += coefficients[i] * X_matrix[key][id]
            i += 1
        if id < 19:
            Y_pred_train.append(y)
        elif id >= 19:
            Y_pred_test.append(y)
    
    plt.scatter(Y_obs_test, Y_pred_test, color='red', label='Zbiór walidacyjny')
    plt.scatter(Y_obs_train, Y_pred_train, color='blue', label='Zbiór uczący')
    plt.plot([Y_vector.min(), Y_vector.max()], [Y_vector.min(), Y_vector.max()], 'k--', lw=2)
    plt.legend()
    plt.xlabel('Wartości eksperymentalne [y_obs]')
    plt.ylabel('Wartości przewidywane [y_pred]')
    plt.title('Wykres y_pred od y_obs')
    plt.show()


def data_loader(path: str):
    df = pd.read_excel(path)
    Y_vector = df[['logK HSA']]
    Y_vector_train = Y_vector.iloc[:19]
    Y_vector_test = Y_vector.iloc[19:]

    X_matrix = df[['logKCTAB', 'CATS3D_00_DD', 'CATS3D_09_AL', 'CATS3D_00_AA']]
    X_matrix_train = X_matrix.iloc[:19]
    X_matrix_test = X_matrix.iloc[19:]
    return Y_vector, X_matrix, Y_vector_train, Y_vector_test, X_matrix_train, X_matrix_test

def corr_matrix(X_matrix):
    corr = X_matrix.corr()
    plt.figure(figsize=(4,4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()
    return corr

def rownanie_modelu(X_matrix, Y_vector):
    model = LinearRegression()
    model.fit(X_matrix, Y_vector)

    intercept = model.intercept_
    coefficients = model.coef_
    coefficients = coefficients[0]


    print(f'\n\nRównanie modelu:\n{intercept[0]:.2f} + {coefficients[0]:.2f}*logKCTAB + {coefficients[1]:.2f}*CATS3D_00_DD + {coefficients[2]:.2f}*CATS3D_09_AL + {coefficients[3]:.2f}*CATS3D_00_AA + ϵ')

    return model

if __name__=='__main__':
    path = 'F:\\vs_code_workspace\\uczenie_maszynowe\\dane_leki.xlsx'           # change path to the file
    Y_vector, X_matrix, Y_obs_train, Y_test, X_train, X_test = data_loader(path)
    print(corr_matrix(X_train))
    model = rownanie_modelu(X_train, Y_obs_train)
    wykres_ypred_yobs(model, X_matrix, Y_vector)
    wykres_Williamsa(X_train, Y_obs_train, X_test, Y_test)
    print(Y_vector.shape[0])