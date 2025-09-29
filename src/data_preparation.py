import pandas as pd

path = r"data\assurance-maladie-68d92978e362f464596651.csv"
content = pd.read_csv(path)

content.columns
content.head()
content.info()

# Doc : Après vérification, j'ai vu que les colonnes 'Sex' (valeurs : male/female), 
# 'Smoker' (valeurs : yes/no) et 'Region' (valeurs : northeast/southeast) 
# doivent être converties en int (0 ou 1) et non laissées en objets (object).

content['sex'] = content['sex'].map({'male':1,'female':0})
content['smoker'] = content['smoker'].map({'yes':1,'no':0})
# content['region'] = content['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})
content = pd.get_dummies(content,columns=['region'])

content['region_northeast'] = content['region_northeast'].astype(int)
content['region_northwest'] = content['region_northwest'].astype(int)
content['region_southeast'] = content['region_southeast'].astype(int)
content['region_southwest'] = content['region_southwest'].astype(int)

# Convert boolean values (True/False) to numeric (1/0) 
# because machine learning models require numeric input to perform calculations.

print(content.info())
print(content.head())
print(content.tail())

print(content[['age', 'bmi', 'children', 'charges']].describe().round(2))

print(content['sex'].value_counts())
print(content['smoker'].value_counts())
print(content[['region_northeast','region_northwest','region_southeast','region_southwest']].sum())

print(content.isnull().sum())
print(content.isnull().any())