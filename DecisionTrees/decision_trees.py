'''
    Zadanie 1. (3 punkty)
    Dane wejściowe – ftalany_klasyfikacja.xlsx zawiera dane dotyczące 32
    ftalanów. Dane są już po autoskalowaniu. Podział na zbiór treningowy i walidacyjny znajdują
    się w poszczególnych arkuszach. Związki są podzielone na dwie kategorie: 1 (trwałość w
    przedziale dni-tygodnie) oraz 2 (tygodnie-miesiące).

    1. Przygotuj macierz korelacji pomiędzy zmiennymi niezależnymi i zmienną zależną.
    2. Sprawdź czy zbiór testowy i treningowy są zbalansowane.
    3. Zbuduj model drzewa klasyfikacyjnego w celu przewidywania tego parametru
    (kategorii trwałości). Oceń zdolności prognostyczne modelu na podstawie macierzy
    pomyłek oraz statystyk: czułości, specyficzności, precyzji, współczynnika F1,
    dokładności oraz zbalansowanego błędu.
    4. Dokonaj optymalizacji parametrów dwiema metodami (z uzasadnieniem ich wyboru)


    Zadanie 2. (2 punkty)
    Dane wejściowe – ftalany.xlsx zawiera dane dotyczące 32 ftalanów. Dane są
    już po autoskalowaniu. Podział na zbiór treningowy i walidacyjny znajdują się w
    poszczególnych arkuszach.

    1. Zbuduj model drzewa regresyjnego, aby przewidzieć stałą szybkości degradacji
    poszczególnych związków.
    2. Oblicz statystyki R2, RMSE, Q2 i RMSEex. Narysuj wykres zależności ypred od yobs,
    oraz wykres słupkowy zależności ypred od yobs.
    3. Dokonaj optymalizacji parametrów dwiema metodami (z uzasadnieniem ich wyboru)
    4. Dokonaj interpretacji uzyskanych wyników
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import balanced_accuracy_score

def zadanie1():
    # Load training and test datasets from Excel file
    X_train = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'X_train')
    y_train = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'y_train')
    X_test = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'X_test')
    y_test = dataLoader('./data/ftalany_klasyfikacja.xlsx', 'y_test')

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
    decisionTreeClassifier(X_train, y_train, X_test, y_test)
    
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
        plt.savefig('./DecisionTrees/macierz_korelacji.png')  # Save the plot as an image
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

def decisionTreeClassifier(
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
    model = DecisionTreeClassifier(random_state=42)  # Initialize the decision tree model with a fixed random state
    model.fit(X_train, y_train)  # Train the model

    y_pred = model.predict(X_test)  # Predict labels for the test set

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Macierz pomyłek:")
    print(cm)

    # Calculate various evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)  # Sensitivity (Recall) for class 1
    specificity = recall_score(y_test, y_pred, average='binary', pos_label=2)  # Specificity for class 2
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)  # Precision for class 1
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)  # F1 score for class 1
    balanced_error = 1 - balanced_accuracy_score(y_test, y_pred)  # Calculate the balanced error rate

    # Print model evaluation statistics
    print("Statystyki modelu:")
    print(f"Dokładność: \t\t{accuracy:.2f}")
    print(f"Czułość (Recall): \t{recall:.2f}")
    print(f"Specyficzność: \t\t{specificity:.2f}")
    print(f"Precyzja: \t\t{precision:.2f}")
    print(f"Współczynnik F1: \t{f1:.2f}")
    print(f"Zbalansowany błąd: \t{balanced_error:.2f}")

def decisionTreeClassifierOptimization(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    '''
    ### Placeholder function for decision tree optimization ###
    This function will be implemented to optimize hyperparameters using two methods.
    '''
    pass

if __name__ == '__main__':
    zadanie1()
