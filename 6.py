import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
df=pd.read_csv("Medical Dataset.csv")
df=df.replace("?",np.nan)
model=BayesianModel([('age','heartdisease'),('sex','heartdisease'),
('exang','heartdisease'),('cp','heartdisease'),
('heartdisease','restecg'),('heartdisease','chol')])
model.fit(df,estimator=MaximumLikelihoodEstimator)
infer=VariableElimination(model)
q=infer.query(variables=['heartdisease'],evidence={'restecg':1})
print(q)






# import numpy as np
# import pandas as pd
# from pgmpy.estimators import MaximumLikelihoodEstimator
# from pgmpy.models import BayesianNetwork
# from pgmpy.inference import VariableElimination

# # Load and preprocess dataset
# heartDisease = pd.read_csv('/content/Medical 6 Dataset.csv').replace('?', np.nan)
# heartDisease[['ca', 'thal']] = heartDisease[['ca', 'thal']].apply(pd.to_numeric, errors='coerce')
# heartDisease.dropna(inplace=True)

# # Display sample instances and datatypes
# print('Sample instances from the dataset are given below')
# print(heartDisease.head())
# print('\nAttributes and datatypes')
# print(heartDisease.dtypes)

# # Define the Bayesian Network structure and learn CPD using Maximum Likelihood Estimation
# model = BayesianNetwork([('age', 'heartdisease'), ('gender', 'heartdisease'), ('exang', 'heartdisease'),
#                          ('cp', 'heartdisease'), ('heartdisease', 'restecg'), ('heartdisease', 'chol')])
# print('\nLearning CPD using Maximum likelihood estimators')
# model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# # Inference with Bayesian Network
# print('\nInferencing with Bayesian Network:')
# HeartDiseasetest = VariableElimination(model)

# # Queries
# print('\n1. Probability of HeartDisease given evidence= restecg')
# print(HeartDiseasetest.query(variables=['heartdisease'], evidence={'restecg': 1}))
# print('\n2. Probability of HeartDisease given evidence= cp')
# print(HeartDiseasetest.query(variables=['heartdisease'], evidence={'cp': 2}))
