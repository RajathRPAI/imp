import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset from CSV
data = pd.read_csv('5ds.csv')

# Assuming the CSV has two columns: 'text' and 'label'
texts = data['text']
labels = data['label']

# Preprocess data: Convert text to numerical feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Optional: Print out predictions for a few test samples
for text, true_label, pred_label in zip(texts[:5], y_test[:5], y_pred[:5]):
    print(f'Text: {text}\nTrue Label: {true_label}, Predicted Label: {pred_label}\n')






# Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.datasets import fetch_20newsgroups

# # Load Data
# # Fetch the 20 newsgroups dataset
# categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
# data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# # Extract features and labels
# X = data.data
# y = data.target

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Convert text data into numerical features using TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words='english')
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# # Initialize and train the Naïve Bayes classifier
# clf = MultinomialNB()
# clf.fit(X_train_vec, y_train)

# # Predict the labels for the test set
# y_pred = clf.predict(X_test_vec)

# # Compute accuracy and classification report
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred, target_names=data.target_names)

# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:")
# print(report)
