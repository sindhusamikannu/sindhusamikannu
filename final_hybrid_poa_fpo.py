import pandas as pd
import numpy as np
import math 
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, roc_curve, precision_recall_curve

# Redirect standard output to a log file
log_file = open('final_hybrid_pelican_fpo_classification_log.txt', 'w')
sys.stdout = log_file

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Preprocess the dataset
X = data.drop('label', axis=1)
y = data['label']

# Encode the labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Pelican Optimization Algorithm for initial feature selection
class PelicanOptimization:
    def __init__(self, num_features, num_pelicans, max_iter):
        self.num_features = num_features
        self.num_pelicans = num_pelicans
        self.max_iter = max_iter

    def initialize_population(self):
        return np.random.randint(2, size=(self.num_pelicans, self.num_features))

    def fitness(self, solution, X, y):
        selected_features = X[:, solution == 1]
        if selected_features.shape[1] == 0:
            return 0
        model = RandomForestClassifier()
        model.fit(selected_features, y)
        predictions = model.predict(selected_features)
        return accuracy_score(y, predictions)

    def optimize(self, X, y):
        population = self.initialize_population()
        best_solution = population[0]
        best_fitness = self.fitness(best_solution, X, y)
        
        for _ in range(self.max_iter):
            new_population = population.copy()
            for i in range(self.num_pelicans):
                new_solution = self.mutate(population[i])
                new_fitness = self.fitness(new_solution, X, y)
                if new_fitness > best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
            population = new_population

        return population, best_solution

    def mutate(self, solution):
        mutation_prob = 0.1
        for i in range(len(solution)):
            if np.random.rand() < mutation_prob:
                solution[i] = 1 - solution[i]
        return solution

# Flower Pollination Algorithm (FPO) for refining feature selection
class FlowerPollinationOptimization:
    def __init__(self, fitness_func, population, p=0.8, max_iter=50):
        self.fitness_func = fitness_func
        self.population = population  # Population from POA
        self.p = p  # Probability switch between global and local pollination
        self.max_iter = max_iter
        self.num_features = population.shape[1]
        self.best_solution = self.population[0].copy()  # Initialize with the first solution in the population
        self.best_fitness = self.fitness_func(self.best_solution)

    def levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.num_features) * sigma
        v = np.random.randn(self.num_features)
        step = u / abs(v) ** (1 / beta)
        return step

    def global_pollination(self, flower):
        if self.best_solution is None:
            self.best_solution = flower.copy()
        new_flower = flower + self.levy_flight() * (self.best_solution - flower)
        return np.clip(new_flower, 0, 1)

    def local_pollination(self, flower, another_flower):
        eps = np.random.random()
        new_flower = flower + eps * (another_flower - flower)
        return np.clip(new_flower, 0, 1)

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(len(self.population)):
                if np.random.random() < self.p:
                    # Global pollination
                    new_solution = self.global_pollination(self.population[i])
                else:
                    # Local pollination
                    j, k = np.random.choice(range(len(self.population)), size=2, replace=False)
                    new_solution = self.local_pollination(self.population[i], self.population[j])
                
                new_solution = np.round(new_solution)
                new_fitness = self.fitness_func(new_solution)
                
                if new_fitness > self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_solution
            
            for i in range(len(self.population)):
                self.population[i] = self.best_solution

        return self.best_solution

# Apply Pelican Optimization for initial feature selection
poa = PelicanOptimization(num_features=X_train.shape[1], num_pelicans=30, max_iter=100)
population, initial_best_solution = poa.optimize(X_train_scaled, y_train)

# Refine the population of POA with Flower Pollination Optimization (FPO)
def fpo_fitness(solution):
    selected_features = X_train_scaled[:, solution == 1]
    if selected_features.shape[1] == 0:
        return 0
    model = RandomForestClassifier()
    model.fit(selected_features, y_train)
    predictions = model.predict(X_test_scaled[:, solution == 1])
    return accuracy_score(y_test, predictions)

fpo = FlowerPollinationOptimization(fitness_func=fpo_fitness, population=population)
best_solution = fpo.optimize()

X_train_selected = X_train_scaled[:, best_solution == 1]
X_test_selected = X_test_scaled[:, best_solution == 1]

# Classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    y_pred_prob = clf.predict_proba(X_test_selected) if hasattr(clf, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\n{name} Classifier Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Cohen's Kappa: {kappa}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    if y_pred_prob is not None:
        auc_roc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        print(f"AUC-ROC: {auc_roc}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve ({name})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'final_poa_fpo_{name}_roc_curve.png')
        plt.close()
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob[:, 1], pos_label=1)
        plt.figure()
        plt.plot(recall_curve, precision_curve, color='green', label=f'Precision-Recall Curve ({name})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {name}')
        plt.legend(loc="lower left")
        plt.savefig(f'final_poa_fpo_{name}_precision_recall_curve.png')
        plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'final_poa_fpo_{name}_confusion_matrix.png')
    plt.close()

    # Save results for comparative plots
    results[name] = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Cohen\'s Kappa': kappa,
        'Matthews Correlation Coefficient': mcc
    }

# Close the log file
log_file.close()

# Comparative Bar Plot
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Cohen\'s Kappa', 'Matthews Correlation Coefficient']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [results[clf][metric] for clf in results], color='orange')
    plt.title(f'Comparative {metric} of Classifiers')
    plt.ylabel(metric)
    plt.xlabel('Classifiers')
    plt.ylim(0, 1)
    plt.savefig(f'final_poa_fpo_comparative_{metric.lower()}.png')
    plt.close()

