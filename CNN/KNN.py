'''
TODO:
    - check what's done, what need updates and what needs to be done
    - make a list of this things acording to .pdf file
    - code review of done things
    - write readme.md file with instructions how to install, setup and use program
    - push to git
    - send email
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
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def prepare_data(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.DataFrame, 
        y_test: pd.DataFrame
    ) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
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
        show_fig: bool = True
    ) -> int:

    if max_k == -1:
        max_k = X_train.shape[0]

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
        n_neighbors: int = 5
) -> KNeighborsRegressor:
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def stats_(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame
) -> list[float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return [rmse, r2]

def y_true_pred(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame
) -> None:
    regression = 0

    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot(y_true, regression, linestyle='--', color='red')
    plt.xlabel('True y values')
    plt.ylabel('Predicted y values')
    plt.grid()
    plt.show()

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