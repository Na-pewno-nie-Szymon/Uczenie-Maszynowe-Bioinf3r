'''
TODO:
    - check what's done, what need updates and what needs to be done
    - make a list of this things acording to .pdf file
    - code review of done things
    - write readme.md file with instructions how to install, setup and use program
    - push to git
    - send email
'''

'''
TODO:
    - []    Describe what n_split is and why it's set to 6
    - []    Describe N-neighbors parameter and implement distance into training
    - []    Calculate Q2 and RMSEex for normal knn
    - []    Calculate R2, RMSE, Q2, RMSEex for model with different destances. Then plot bar cahrt of diferent destances
    - []    Przygotuj analizę ważonego modelu KNN oraz oblicz statystyki R2, RMSE, Q2, RMSEex + wykres y_pred od y_true dla ważonego modelu KNN
    - []    Statystyki R2, RMSE, Q2, RMSEex dla ważonego modelu z różnymi dystansami:
            - Euklides
            - Manhattan
            - Czebyszew
            - Canberra
            Następnie przygotuj wykres słupkowy ze statystykami odległości 
'''

import numpy as np
import pandas as pd
from math import nan
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data(
        file_path: str, 
        sheet_name: str
    ) -> pd.DataFrame:
    '''
    ### Loads data from excel file
    @param file_path path to file 
    @param sheet_name name of int file sheet from witch you want o import data
    @return data in pandas.DataFrame format 
    '''
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def prepare_data(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.DataFrame, 
        y_test: pd.DataFrame
    ) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    ### Normalises data and adjusts column names
    @param X_train Trening matrix
    @param X_test Validation matrix
    @param y_train Trening vector
    @param y_test Validation vector
    @return return autoscaled data, with proper column names
    '''
    scaler = StandardScaler()

    X_test = X_test.rename(columns={'qc': 'qc-'})
    X_train = X_train.rename(columns={'qc': 'qc-'})

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    return [X_train_scaled, X_test_scaled, y_test.values.ravel(), y_train.values.ravel()]

def optimal_K_finder(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        max_k: int = -1, 
        n_splits: int = 6,
        show_fig: bool = True,
        save_fig: bool = False 
    ) -> int:
    '''
    ### Function that finds k parameter that fits the best for the model using KFold
    @param X_train Trening matrix
    @param y_train Trening vector
    @param max_k maximum k, if not declared max_k takes size of X_train - 1
    @param n_splits 
    @param show_fig False for not showing figure 
    @param save_fig Decides if figure will be saved 
    @return Returns best fitting k param
    @todo describe what n_splits is and why if not defined it's equal to 6  
    '''

    if max_k == -1:
        max_k = X_train.shape[0] -1

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    rmse_values: list[float] = []
    k_values: list[int] = []

    k = int(np.sqrt(X_train.shape[0]))
    rmse = 0

    while k < max_k:
        knn = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())

        if rmse >= 0:
            print(f'n_neighbors: {k},\tRMSE: {rmse}')
            rmse_values.append(rmse)
            k_values.append(k)
            k += 1
        else:
            print(f'At n_neighbors: {k}, RMSE value was lower than 0 (RMSE={rmse})!')
            break
    
    optimal_k = np.argmin(rmse_values) + 1
    
    if show_fig:
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, rmse_values, marker='o', linestyle='--', color='b')
        plt.xlabel('Neighbors # (k)')
        plt.ylabel('RMSE')
        plt.title('RMSE at each neighbor #')
        plt.xticks(k_values)
        plt.grid()
        plt.show()

    return optimal_k

def train_knn(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_neighbors: int = 5,
        distance: str = ""
) -> KNeighborsRegressor:
    '''
    ### Function that trains K-Nearest-Neighbors model
    @param X_train Training matrix
    @param y_train Training vector
    @param n_neighbors 
    @distance defines distance used to train model
    @returns Trained model
    @todo Describe N-neighbors parameter and implement distance into training
    '''
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def stats_(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame
) -> list[float, float]:
    '''
    ### Function that calculates r_sqared and RMSE for predicted values
    @param y_true Validation vector
    @param y_pred Predicted vector
    @return R sqared and RMSE values
    @todo calculate q2 and RMSEex
    '''
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    q2 = 0
    RMSEex = 0
    return [RMSE, r2, q2, RMSEex]


def y_true_pred(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        save_fig: bool = False
) -> None:
    '''
    ### Functiond that visualise accuracy of prediction model
    @param y_true Validation vector
    @param y_pred Predicted vector
    @param save_fig Decides if figure should be saved if current directory
    '''

    regression = 0

    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot(y_true, regression, linestyle='--', color='red')
    plt.xlabel('True y values')
    plt.ylabel('Predicted y values')
    plt.grid()
    plt.show()

    if save_fig:
        plt.savefig()

    return None

if __name__=='__main__':
    FILE_PATH = './data/ftalany.xlsx'

    X_train = load_data(FILE_PATH, sheet_name='')
    X_test = load_data(FILE_PATH, sheet_name='')
    y_train = load_data(FILE_PATH, sheet_name='')
    y_test = load_data(FILE_PATH, sheet_name='')

    optimal_k = optimal_K_finder(X_train, y_train)
    print(f'Optimal neighbor #: {optimal_k}')

    model = train_knn(X_train, y_train, n_neighbors=optimal_k)

    y_pred = model.predict(X_test)

    _stats = stats_(y_test, y_pred)
    rmse, r2 = _stats[0], _stats[1]
    print(f'R²: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    y_true_pred(y_test, y_pred)