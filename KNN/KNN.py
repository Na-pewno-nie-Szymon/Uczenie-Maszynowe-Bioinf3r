import numpy as np
import pandas as pd
from math import nan
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, DistanceMetric
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
    print(type(data))
    return data

def std_scaler_and_preprocessor(
        matrix: pd.DataFrame, 
        wrong_name: list = [], 
        new_name: list = [], 
    ) -> pd.DataFrame:
    '''
    ### Standarises given matrix and hanges names of declared columns
    @param matrix Matrix with wrong column name
    @param wrong_name Wrong column name
    @param new_name New proper name for the column
    @return Normalised matrix with proper column naming
    '''
    change = (len(wrong_name) == len(new_name)) and len(wrong_name) != 0

    if change:
        columns = dict(zip(wrong_name, new_name))
        matrix = matrix.rename(columns=columns)

    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    return matrix

def optimal_Knum_model(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        max_k: int = -1, 
        n_splits: int = 3,
        show_fig: bool = True,
        save_fig: bool = False, 
        distance: str = '',
        weight: str = 'uniform'
    ) -> list[int, KNeighborsRegressor]:
    '''
    ### Function that finds k parameter that fits the best for the model using KFold
    @param X_train Trening matrix
    @param y_train Trening vector
    @param max_k maximum k, if not declared max_k takes size of X_train - 1
    @param n_splits number od consecutive folds 
    @param show_fig False for not showing figure 
    @param save_fig Decides if figure will be saved
    @param distance choose avaliable distance
    @param weight Defines weight used to train model 
    @return Returns best fitting k param
    '''

    possible_distances = ['Euclidean', 'Manhattan', 'Canberra', 'Chebyshev', '']
    if distance not in possible_distances:
        raise ValueError(f'Distance metric {distance}, doesn\'t exist!')

    if max_k == -1:
        max_k = int(X_train.shape[0] * 0.75)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    rmse_values: list[float] = []
    k_values: list[int] = []

    k = 1
    lowest_rmse = 1


    best_model = None

    while k < max_k:
        if distance == '':
            knn = train_knn(X_train, y_train, n_neighbors=k)
        else:
            knn = train_knn(X_train, y_train, n_neighbors=k, distance=distance, weight=weight)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())

        if rmse >= 0:
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                best_model = knn
                best_k = k

            print(f'n_neighbors: {k}, \tRMSE: {rmse}')
            rmse_values.append(rmse)
            k_values.append(k)
            k += 1
        else:
            print(f'At n_neighbors: {k}, RMSE value was lower than 0 (RMSE={rmse})!')
            break
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, rmse_values, marker='o', linestyle='--', color='b')
    plt.xlabel('Neighbors # (k)')
    plt.ylabel('RMSE')
    plt.title(f'{distance} RMSE at each neighbor #')
    plt.xticks(k_values)
    plt.grid()
    if save_fig:
        plt.savefig(f'./figures/{distance}_weight_{weight}_KNN_model.png')
    if show_fig:
        plt.show()
    else:
        plt.pause

    return [best_k, best_model]

def train_knn(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_neighbors: int = 5,
        distance: str = "",
        weight: str = 'uniform'
) -> KNeighborsRegressor:
    '''
    ### Function that trains K-Nearest-Neighbors model
    @param X_train Training matrix
    @param y_train Training vector
    @param n_neighbors Number of neighbors
    @param distance Defines distance used to train model
    @param weight Defines weight used to train model
    @returns Trained model
    '''
    possible_distances = ['Euclidean', 'Manhattan', 'Canberra', 'Chebyshev', '']
    if distance not in possible_distances:
        raise ValueError(f'Distance metric {distance}, doesn\'t exist!')

    if distance == possible_distances[4]    : knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='minkowski', weights=weight)
    elif distance == possible_distances[0]  : knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=DistanceMetric.get_metric('euclidean'), weights=weight)
    elif distance == possible_distances[1]  : knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=DistanceMetric.get_metric('manhattan'), weights=weight)
    elif distance == possible_distances[2]  : knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=DistanceMetric.get_metric('canberra'), weights=weight)
    elif distance == possible_distances[3]  : knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=DistanceMetric.get_metric('chebyshev'), weights=weight)

    knn.fit(X_train, y_train)

    return knn

def stats_(
        y_train_true: pd.DataFrame,
        y_train_pred: pd.DataFrame,
        y_val_pred: pd.DataFrame,
        y_val_true: pd.DataFrame
) -> list[float, float, float, float]:
    '''
    ### Function that calculates r_sqared and RMSE for predicted values
    @param y_train_true Training true-value vector
    @param y_train_pred Training pred-value vector
    @param y_val_true Validation true-value vector
    @param y_val_pred Validation pred-value vector
    @return R sqared and RMSE values
    '''
    
    r2 = r2_score(y_train_true, y_train_pred)
    RMSE = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    
    y_val_mean = np.mean(y_val_true)
    q2 = 1 - (np.sum((y_val_true - y_val_pred)**2) / np.sum((y_val_true - y_val_mean)**2))
    RMSEex = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    return [RMSE, r2, q2, RMSEex]


def y_true_pred(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        distance: str,
        save_fig: bool = False,
        weighted: bool = False
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

    if save_fig and weighted:
        plt.savefig(f'./figures/weighted_{distance}_y_true_pred.png')
    elif save_fig and not weighted:
        plt.savefig(f'./figures/{distance}_y_true_pred.png')
    
    plt.show()
    return None

def y_true_pred_bar_chart(
        y_true: pd.DataFrame, 
        y_pred: pd.DataFrame,
        weighted: bool,
        distance: str,
        show_fig: bool = False,
        save_fig: bool = False
    ) -> None:
    bar_width = 0.15
    fig = plt.subplots(figsize=(12, 8))
    
    br1 = np.arange(len(y_true)) 
    br2 = [x + bar_width for x in br1] 
    
    plt.bar(br1, y_true.ravel(), color ='r', width = bar_width, 
            edgecolor ='grey', label ='y true') 
    plt.bar(br2, y_pred.ravel(), color ='g', width = bar_width, 
            edgecolor ='grey', label ='y pred') 
    
    plt.xlabel('y indexes', fontweight ='bold', fontsize = 15) 
    plt.ylabel('y value', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + bar_width for r in range(len(y_true))], list(range(len(y_true))))
    
    plt.title('y value comparisson')
    plt.legend()

    if save_fig and weighted:
        plt.savefig(f'./figures/weighted_{distance}_y_values.png')
    elif save_fig and not weighted:
        plt.savefig(f'./figures/{distance}_y_values.png')

    if show_fig:
        plt.show()


def zadanie1(file_path: str) -> None:
    X_train = std_scaler_and_preprocessor(load_data(file_path, sheet_name='X_train'))
    X_test = std_scaler_and_preprocessor(load_data(file_path, sheet_name='X_test'), ['qc'], ['qc-'])
    y_train = std_scaler_and_preprocessor(load_data(file_path, sheet_name='y_train'))
    y_test = std_scaler_and_preprocessor(load_data(file_path, sheet_name='y_test'))

    optimal_k, model = optimal_Knum_model(X_train, y_train)
    print(f'Optimal neighbor #: {optimal_k}')

    y_val_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    _stats = stats_(y_train, y_train_pred, y_test, y_val_pred)
    rmse, r2, q2, rmse_ex = _stats[0], _stats[1], _stats[2], _stats[3]
    print(f'R²: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'Q2: {q2:.4f}')
    print(f'RMSEex: {rmse_ex:.4f}')
    y_true_pred(y_test, y_val_pred, save_fig=True, distance='Minkowski')

    possible_distances = ['Euclidean', 'Manhattan', 'Canberra', 'Chebyshev']
    r2 = []
    q2 = []
    rmse = []
    rmse_ex = []
    optimal_k = []

    for dist in possible_distances:
        optimal_k_, model = optimal_Knum_model(X_train, y_train, distance=dist, show_fig=False, save_fig=True)
        y_val_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        _stats = stats_(y_train, y_train_pred, y_test, y_val_pred)
        r2.append(_stats[1])
        q2.append(_stats[2])
        rmse.append(_stats[0])
        rmse_ex.append(_stats[3])
        optimal_k.append(optimal_k_)
        y_true_pred(y_test, y_val_pred, save_fig=True, distance=dist)
        y_true_pred_bar_chart(y_test, y_val_pred, weighted=False, show_fig=False, save_fig=True, distance=dist)

    bar_width = 0.15
    fig = plt.subplots(figsize=(12, 8))
    
    br1 = np.arange(len(r2)) 
    br2 = [x + bar_width for x in br1] 
    br3 = [x + bar_width + 0.05 for x in br2] 
    br4 = [x + bar_width for x in br3]

    plt.bar(br1, r2, color ='r', width = bar_width, 
            edgecolor ='grey', label ='R2') 
    #plt.text(br1, r2, str(r2), color='black', fontsize=10)

    plt.bar(br2, rmse, color ='g', width = bar_width, 
            edgecolor ='grey', label ='RMSE') 
    #plt.text(br2, rmse, str(rmse), color='black', fontsize=10)
    
    plt.bar(br3, q2, color ='b', width = bar_width, 
            edgecolor ='grey', label ='q2')
    #plt.text(br3, q2, str(q2), color='black', fontsize=10)

    plt.bar(br4, rmse_ex, color ='pink', width = bar_width, 
            edgecolor ='grey', label ='RMSEex')
    #plt.text(br4, rmse_ex, str(rmse_ex), color='black', fontsize=10)

    
    plt.xlabel('Distance metric', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Value', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + bar_width for r in range(len(r2))], 
            ['Euclidean', 'Manhattan', 'Canberra', 'Chebyshev'])
    
    plt.title('Different distance metrics')
    plt.legend()
    plt.savefig('./figures/diferent_distance_stat_comparisson.png')
    plt.show()

def zadanie2(file_path: str) -> None:
    X_train = std_scaler_and_preprocessor(load_data(file_path, sheet_name='X_train'))
    X_test = std_scaler_and_preprocessor(load_data(file_path, sheet_name='X_test'), ['qc'], ['qc-'])
    y_train = std_scaler_and_preprocessor(load_data(file_path, sheet_name='y_train'))
    y_test = std_scaler_and_preprocessor(load_data(file_path, sheet_name='y_test'))

    optimal_k, model = optimal_Knum_model(X_train, y_train)
    print(f'Optimal neighbor #: {optimal_k}')

    y_val_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    _stats = stats_(y_train, y_train_pred, y_test, y_val_pred)
    rmse, r2, q2, rmse_ex = _stats[0], _stats[1], _stats[2], _stats[3]
    print(f'R²: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'Q2: {q2:.4f}')
    print(f'RMSEex: {rmse_ex:.4f}')
    y_true_pred(y_test, y_val_pred, save_fig=True, distance='Minkowski', weighted=True)

    possible_distances = ['Euclidean', 'Manhattan', 'Canberra', 'Chebyshev']
    r2 = []
    q2 = []
    rmse = []
    rmse_ex = []
    optimal_k = []

    for dist in possible_distances:
        optimal_k_, model = optimal_Knum_model(X_train, y_train, distance=dist, show_fig=False, save_fig=True, weight='distance')
        y_val_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        _stats = stats_(y_train, y_train_pred, y_test, y_val_pred)
        r2.append(_stats[1])
        q2.append(_stats[2])
        rmse.append(_stats[0])
        rmse_ex.append(_stats[3])
        optimal_k.append(optimal_k_)
        y_true_pred(y_test, y_val_pred, save_fig=True, distance=dist, weighted=True)
        y_true_pred_bar_chart(y_test, y_val_pred, weighted=True, show_fig=False, save_fig=True, distance=dist)

    bar_width = 0.15
    fig = plt.subplots(figsize=(12, 8))
    
    br1 = np.arange(len(r2)) 
    br2 = [x + bar_width for x in br1] 
    br3 = [x + bar_width + 0.05 for x in br2] 
    br4 = [x + bar_width for x in br3]

    plt.bar(br1, r2, color ='r', width = bar_width, 
            edgecolor ='grey', label ='R2') 
    #plt.text(br1, r2, str(r2), color='black', fontsize=10)

    plt.bar(br2, rmse, color ='g', width = bar_width, 
            edgecolor ='grey', label ='RMSE') 
    #plt.text(br2, rmse, str(rmse), color='black', fontsize=10)
    
    plt.bar(br3, q2, color ='b', width = bar_width, 
            edgecolor ='grey', label ='q2')
    #plt.text(br3, q2, str(q2), color='black', fontsize=10)

    plt.bar(br4, rmse_ex, color ='pink', width = bar_width, 
            edgecolor ='grey', label ='RMSEex')
    #plt.text(br4, rmse_ex, str(rmse_ex), color='black', fontsize=10)
    
    plt.xlabel('Distance metric', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Value', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + bar_width for r in range(len(r2))], 
            ['Euclidean', 'Manhattan', 'Canberra', 'Chebyshev'])
    
    plt.title('Different distance metrics')
    plt.legend()
    plt.savefig('./figures/diferent_weighted_distance_stat_comparisson.png')
    plt.show()
    



if __name__=='__main__':
    FILE_PATH = './data/ftalany.xlsx'

    zadanie1(FILE_PATH)
    zadanie2(FILE_PATH)