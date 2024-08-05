
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def build_decision_tree(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(X, y)
    return tree

def print_tree(tree, feature_names):
    tree_rules = export_text(tree, feature_names=feature_names)
    print(tree_rules)

# Example usage:
data = read_csv('3ds.csv')
tree = build_decision_tree(data)
print_tree(tree, data.columns[:-1])

