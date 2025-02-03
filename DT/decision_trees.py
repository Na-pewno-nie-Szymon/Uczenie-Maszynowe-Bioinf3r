'''
    ### Zadanie 1
    [x] wczytaj dane treningowe i testowe z pliku ftalany_klasyfikacja.xlsx
    [x] dla pewności autoskaluj dane
    [x] wygeneruj macierz korelacji pomiędzy zmiennymi niezależnymi i zmienną zależną
    [x] sprawdź czy zbiory treningowy i testowy są zbalansowane
    [x] zbuduj model drzewa klasyfikacyjnego
    [x] ocen zdolności prognostyczne modelu na podstawie macierzy pomyłek oraz statystyk: 
        [x] czułości
        [x] specyficzności
        [x] precyzji
        [x] współczynnika F1
        [x] dokładności
        [x] zbalansowanego błędu
    [x] wyświetl drzewo decyzyjne
    [x] dokonaj optymalizacji parametrów modelu drzewa klasyfikacyjnego

    ### Zadanie 2
    [x] wczytaj dane treningowe i testowe z pliku ftalany.xlsx
    [x] dla pewności autoskaluj dane
    [x] zbuduj model drzewa regresyjnego
    [x] oblicz statystyki R2, RMSE, Q2 i RMSEex
    [x] narysuj wykres zależności ypred od yobs
    [x] narysuj wykres słupkowy zależności ypred od yobs
    [x] dokonaj optymalizacji parametrów modelu drzewa regresyjnego
    [ ] dokonaj interpretacji uzyskanych wyników
'''

'''
    DEBUG CODE:
    ValueError: Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.

'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, r2_score


def zadanie2():
    FILE_PATH = '~/Downloads/ftalany(1).xlsx'

    X_train = autoscaleData(dataLoader(FILE_PATH, 'X_train'))
    y_train = autoscaleData(dataLoader(FILE_PATH, 'y_train'))
    X_test = autoscaleData(dataLoader(FILE_PATH, 'X_test'))
    y_test = autoscaleData(dataLoader(FILE_PATH, 'y_test'))

    # 1. Create a correlation matrix between independent and dependent variables
    Xy_matrix = pd.concat([X_train, y_train], axis=1)  # Combine features and labels into one DataFrame
    corr_matrix = correlationMatrix(Xy_matrix)  # Generate the correlation matrix
    print('Correlation matrix:')
    print(corr_matrix, '\n')

    # 2. Check if the training and test datasets are balanced
    check_balance(y_train, 'treningowy')  # Check balance for training set
    print()
    check_balance(y_test, 'testowy')  # Check balance for test set
    print()

    # 3. Build and evaluate a decision tree regressor model
    DT_model_regresja(X_train, y_train, X_test, y_test)


    # 4. Optimize the regressor using two methods
    regression_DT_optimization(X_train, y_train, X_test, y_test)



def regression_DT_optimization(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
        ) -> None:
    '''
    ### Function to optimize the decision tree regressor model ###
    @param X_train: training set of independent variables
    @param y_train: training set labels
    @param X_test: test set of independent variables
    @param y_test: test set labels
    @return None
    '''
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 1) Manual search
    best_model_manual = None
    best_score_manual = float('-inf')
    best_params_manual = {}


    for max_depth in param_grid["max_depth"]:
        for min_samples_split in param_grid["min_samples_split"]:
            for min_samples_leaf in param_grid["min_samples_leaf"]:
                model = DecisionTreeRegressor(
                    random_state=42,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf
                )
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))

                if score > best_score_manual:
                    best_score_manual = score
                    best_params_manual = {
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf
                    }

    print(f"\nNajlepsze parametry (Manual search):\n {best_params_manual}")
    print(f"R2 (Manual search): {best_score_manual}")

    # Losowe przeszukiwanie hiperparametrów
    random_search = RandomizedSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="neg_mean_squared_error", # Zmiana z R2 na neg_mean_squarred_error bo wychodza absurdalne wartosci R2
    random_state=42
    )
    random_search.fit(X_train, y_train)

    best_params_random = random_search.best_params_
    best_score_random = random_search.best_score_

    print(f"\nNajlepsze parametry (Randomized search):\n{best_params_random}")
    print(f"R2 (Randomized search): {best_score_random}")


def y_pred_obs_bar_plot(
        y_test: pd.Series,
        y_pred: pd.Series
        ) -> None:
    '''
    ### Function to plot the relationship between predicted and observed values as a bar plot ###
    @param y_test: observed values
    @param y_pred: predicted values
    @return None
    '''
    plt.figure(figsize=(8, 6))
    plt.bar(y_test.values.ravel(), y_pred.ravel(), color='blue', alpha=0.5)
    plt.xlabel('Obserwowane wartości')
    plt.ylabel('Przewidziane wartości')
    plt.title('Zależność między obserwowanymi i przewidywanymi wartościami')
    plt.savefig('./figures/DT/y_pred_obs_bar_plot.png')  # Save the plot as an image
    plt.show()

def y_pred_obs_plot(
        y_test: pd.Series,
        y_pred: pd.Series
        ) -> None:
    '''
    ### Function to plot the relationship between predicted and observed values ###
    @param y_test: observed values
    @param y_pred: predicted values
    @return None
    
    '''
    coef = np.polyfit(y_test.values.ravel(), y_pred.ravel(), 1)
    poly1d_fn = np.poly1d(coef) 
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot(y_test, poly1d_fn(y_test), linestyle='--', color='red')
    plt.xlabel('Obserwowane wartości')
    plt.ylabel('Przewidziane wartości')
    plt.title('Zależność między obserwowanymi i przewidywanymi wartościami')
    plt.savefig('./figures/DT/y_pred_obs_plot.png')  # Save the plot as an image
    plt.show()

def DT_model_regresja(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
        ) -> None:
    '''
    ### Function to build and evaluate a decision tree regressor ###
    @param X_train: training set of independent variables
    @param y_train: training set labels
    @param X_test: test set of independent variables
    @param y_test: test set labels
    @return None
    '''
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=5,            # Ograniczenie głębokości drzewa
        min_samples_split=10,   # Minimalna liczba próbek do podziału
        min_samples_leaf=5      # Minimalna liczba próbek w liściu
    )
    model.fit(X_train, y_train)

    plot_tree(
        model,
        filled=True,
        feature_names=X_train.columns,
        class_names=['1', '2']
    )  # Plot the decision tree

    y_pred = model.predict(X_test)  # Predict labels for the test set
    print(y_pred)

    plt.show()

    stats_model(X_train, y_train, X_test, y_test, y_test_pred=y_pred, y_train_pred=model.predict(X_train))

    y_pred_obs_plot(y_test, y_pred)
    y_pred_obs_bar_plot(y_test, y_pred)

def stats_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_test_pred: pd.Series,
        y_train_pred: pd.Series
        ) -> None:
    '''
    ### Function to build and evaluate a decision tree regressor ###
    @param X_train: training set of independent variables
    @param y_train: training set labels
    @param X_test: test set of independent variables
    @param y_test: test set labels
    @return None
    '''
    print(len(y_train), len(y_train_pred))
    r2 = r2_score(y_train, y_train_pred)
    rmse = root_mean_squared_error(y_train, y_train_pred)
    q2 = r2_score(y_test, y_test_pred)
    rmseex = root_mean_squared_error(y_test, y_test_pred)

    print(f'R2: {r2}')
    print(f'RMSE: {rmse}')
    print(f'Q2: {q2}')
    print(f'RMSEex: {rmseex}')
    

def zadanie1():
    # Load training and test datasets from Excel file
    FILE_PATH = 'data/ftalany_klasyfikacja.xlsx'

    X_train = autoscaleData(dataLoader(FILE_PATH, 'X_train'))
    y_train = autoscaleData(dataLoader(FILE_PATH, 'y_train'))
    X_test = autoscaleData(dataLoader(FILE_PATH, 'X_test'))
    y_test = autoscaleData(dataLoader(FILE_PATH, 'y_test'))

    # 1. Create a correlation matrix between independent and dependent variables
    Xy_matrix = pd.concat([X_train, y_train], axis=1)  # Combine features and labels into one DataFrame
    corr_matrix = correlationMatrix(Xy_matrix)  # Generate the correlation matrix
    print('Correlation matrix:')
    print(corr_matrix, '\n')

    # 2. Check if the training and test datasets are balanced
    check_balance(y_train, 'treningowy')  # Check balance for training set
    print()
    check_balance(y_test, 'testowy')  # Check balance for test set
    print()

    # 3. Build and evaluate a decision tree classifier model
    DT_model(X_train, y_train, X_test, y_test)
    
    # 4. Optimize the classifier using two methods
    decisionTreeClassifierOptimization(X_train, y_train, X_test, y_test)

def dataLoader(
        path: str, 
        sheet_name: str
    ) -> pd.DataFrame:
    '''
    ### Function to load data from an Excel sheet ###
    @param path path to the Excel file
    @param sheet_name name of the sheet to load
    @param return DataFrame with data from the specified sheet
    '''
    data = pd.read_excel(path, sheet_name=sheet_name)  # Read the specified sheet from the Excel file
    return data

def autoscaleData(matrix: pd.DataFrame) -> pd.DataFrame:
    '''
    ### Function to autoscale data ###
    @param matrix: DataFrame with the data
    @return DataFrame with autoscaled data
    '''
    return (matrix - matrix.mean()) / matrix.std()  # Autoscale the data

def correlationMatrix(
        data: pd.DataFrame, 
        visualise: bool = True
    ) -> pd.DataFrame:
    '''
    ### Function to create and visualize a correlation matrix ###
    @param data: DataFrame with the data
    @param visualise: flag to determine whether to display the heatmap
    @return DataFrame with the correlation matrix
    '''
    correlation_matrix = data.corr()  # Compute the correlation matrix

    if visualise:
        # Create a heatmap visualization of the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Macierz Korelacji')
        plt.savefig('./figures/DT/macierz_korelacji.png')  # Save the plot as an image
        plt.show()
    
    return correlation_matrix

def check_balance(
        y: pd.Series, 
        set_name: str
    ) -> None:
    '''
    ### Function to check if a dataset is balanced ###
    @param y: labels of the dataset
    @param set_name: name of the dataset (e.g., 'training' or 'test')
    @return None
    '''
    counts = y.value_counts(normalize=True)  # Calculate class distribution as percentages
    print(f"Balans dla {set_name}:")
    print(counts)
    if counts.min() < 0.4:  # Example threshold to flag imbalance
        print(f"Uwaga: Zbiór {set_name} jest niezbalansowany!")
    else:
        print(f"Zbiór {set_name} jest zbalansowany.")

def DT_model(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> None:
    '''
    ### Function to build and evaluate a decision tree classifier ###
    @param X_train: training set of independent variables
    @param y_train: training set labels
    @param X_test: test set of independent variables
    @param y_test: test set labels
    @return None
    '''
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    model = DecisionTreeClassifier(random_state=42, splitter='best')  # Initialize the decision tree model with a fixed random state
    model.fit(X_train, y_train)  # Train the model

    plot_tree(
        model, 
        filled=True, 
        feature_names=X_train.columns, 
        class_names=['1', '2']
    )  # Plot the decision tree

    y_pred = model.predict(X_test)  # Predict labels for the test set
    
    plt.show()

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Macierz pomyłek:")
    print(cm)

    evaluate_model(y_test, y_pred)

def decisionTreeClassifierOptimization(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> None:
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)


    # Przeszukiwanie siatki hiperparametrów
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    print(f"Najlepsze parametry z Grid Search: {grid_search.best_params_}")
    best_model_grid = grid_search.best_estimator_

    # Prognozowanie i ocena dla najlepszego modelu
    y_pred_grid = best_model_grid.predict(X_test)
    print("Wyniki dla modelu po Grid Search:")
    evaluate_model(y_test, y_pred_grid)

    # Losowe przeszukiwanie hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print(f"Najlepsze parametry z Random Search: {random_search.best_params_}")
    best_model_random = random_search.best_estimator_

    # Prognozowanie i ocena dla najlepszego modelu
    y_pred_random = best_model_random.predict(X_test)
    print("Wyniki dla modelu po Random Search:")
    evaluate_model(y_test, y_pred_random)

def evaluate_model(
        y_test: pd.Series, 
        y_pred: pd.Series
    ) -> None:
    '''
    Funkcja pomocnicza do wyświetlenia statystyk modelu.
    '''
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=-1)
    specificity = recall_score(y_test, y_pred, average='binary', pos_label=0)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=-1)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=-1)
    balanced_error = 1 - balanced_accuracy_score(y_test, y_pred)

    print(f"Dokładność: \t\t{accuracy:.2f}")
    print(f"Czułość (Recall): \t{recall:.2f}")
    print(f"Specyficzność: \t\t{specificity:.2f}")
    print(f"Precyzja: \t\t{precision:.2f}")
    print(f"Współczynnik F1: \t{f1:.2f}")
    print(f"Zbalansowany błąd: \t{balanced_error:.2f}")

if __name__ == '__main__':
    zadanie1()
    zadanie2()
