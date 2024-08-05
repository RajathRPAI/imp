import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def knn_classifier(data, k):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    return accuracy_score(y_test, predictions), predictions

# Example usage:
data = read_csv('8ds.csv')
accuracy, predictions = knn_classifier(data, 3)
print("Accuracy:", accuracy)





# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# def read_csv(file_path):
#     data = pd.read_csv(file_path)
#     return data

# def knn_classifier(data, k):
#     X = data.iloc[:, :-1]  # Features
#     y = data.iloc[:, -1]   # Labels
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, predictions)
    
#     # Print correct and wrong predictions
#     results = pd.DataFrame({
#         'Actual': y_test,
#         'Predicted': predictions,
#         'Correct': y_test == predictions
#     })
    
#     print("Accuracy:", accuracy)
#     print("\nPrediction Results:")
#     print(results)

# # Example usage:
# data = read_csv('8ds.csv')
# knn_classifier(data, 3)
