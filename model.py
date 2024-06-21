# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import joblib
from sklearn.impute import SimpleImputer

# Function to load the dataset
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=column_names)
    return data

# Function to preprocess the data
def preprocess_data(data):
    X = data.drop('Outcome', axis=1)  # Separate features from the target variable
    y = data['Outcome']  # Target variable

    # Handle missing values using median imputation
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, imputer

# Function to train the model
def train_model(X_train, y_train):
    # Define parameter distribution for Randomized Search
    param_dist = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(2, 10),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    }

    xgb_model = XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss')
    
    # Perform Randomized Search
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=100, scoring='accuracy', cv=5, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_

    # Define parameter grid for Grid Search based on best parameters from Randomized Search
    param_grid = {
        'n_estimators': [best_params['n_estimators'] - 10, best_params['n_estimators'], best_params['n_estimators'] + 10],
        'learning_rate': [best_params['learning_rate'] - 0.01, best_params['learning_rate'], best_params['learning_rate'] + 0.01],
        'max_depth': [best_params['max_depth'] - 1, best_params['max_depth'], best_params['max_depth'] + 1],
        'min_child_weight': [best_params['min_child_weight'] - 1, best_params['min_child_weight'], best_params['min_child_weight'] + 1],
        'subsample': [max(0.1, best_params['subsample'] - 0.1), best_params['subsample'], min(1.0, best_params['subsample'] + 0.1)],
        'colsample_bytree': [max(0.1, best_params['colsample_bytree'] - 0.1), best_params['colsample_bytree'], min(1.0, best_params['colsample_bytree'] + 0.1)]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator(), grid_search.best_params_

# Function to evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)  # Predictions on the training set
    y_test_pred = model.predict(X_test)  # Predictions on the test set

    training_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    cv_scores = cross_val_score(model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), cv=10, scoring='accuracy')

    return {
        "training_accuracy": training_accuracy,
        "test_accuracy": test_accuracy,
        "conf_matrix": conf_matrix,
        "class_report": class_report,
        "cv_accuracy_mean": np.mean(cv_scores),
        "cv_accuracy_std": np.std(cv_scores)
    }

# Function to plot the learning curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Plot the learning curve with shaded areas representing the standard deviation
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Function to save the model, scaler, and imputer
def save_model(model, scaler, imputer, filepath="model.pkl"):
    with open(filepath, "wb") as f:
        joblib.dump((model, scaler, imputer), f)

# Function to load the model, scaler, and imputer
def load_model(filepath="model.pkl"):
    with open(filepath, "rb") as f:
        model, scaler, imputer = joblib.load(f)
    return model, scaler, imputer

# Main function to execute the workflow
if __name__ == "__main__":
    data = load_data()  # Load the dataset
    X_train, X_test, y_train, y_test, scaler, imputer = preprocess_data(data)  # Preprocess the data
    model, best_params = train_model(X_train, y_train)  # Train the model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)  # Evaluate the model

    # Print evaluation results
    print(f"Best Parameters: {best_params}")
    print(f"Training Accuracy: {metrics['training_accuracy']:.2f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2f}")
    print("Confusion Matrix:")
    print(metrics['conf_matrix'])
    print("Classification Report:")
    print(metrics['class_report'])
    print(f"Cross-validation Accuracy: {metrics['cv_accuracy_mean']:.2f} ± {metrics['cv_accuracy_std']:.2f}")

    save_model(model, scaler, imputer)  # Save the model, scaler, and imputer

    # Plot learning curve
    plt = plot_learning_curve(model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
    plt.show()
