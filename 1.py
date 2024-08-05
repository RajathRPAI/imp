import csv

def read_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def find_s(data):
    hypothesis = data[0][:-1]
    for instance in data:
        if instance[-1] == 'yes': 
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'
                    
    return hypothesis

# Example usage:
data = read_csv("  ")
hypothesis = find_s(data)
print("Final Hypothesis:", hypothesis)




# import pandas as pd
# import numpy as np
# data = pd.read_csv("Finds.csv")
# attribute = np.array(data)[:, :-1]
# target = np.array(data)[:, -1]
# print(attribute)
# print(target)

# def train(att, tar):
#     specific_h = None

#     for i, val in enumerate(tar):
#         if val == 'yes':
#             specific_h = att[i].copy()
#             print(f"Initial specific hypothesis (first 'yes' instance): {specific_h}")
#             break

#     for i, val in enumerate(att):
#         if tar[i] == 'yes':
#             print(f"\nConsidering instance {i + 1}: {val}")
#             for x in range(len(specific_h)):
#                 if val[x] != specific_h[x]:
#                     specific_h[x] = '?'
#             print(f"Updated specific hypothesis: {specific_h}")

#     return specific_h

# specific_hypothesis = train(attribute, target)
# print("\nFinal specific hypothesis:", specific_hypothesis)
