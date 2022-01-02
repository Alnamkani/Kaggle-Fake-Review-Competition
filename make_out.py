import pandas as pd

data = pd.read_csv('out')
print(len(data))
if len(data) != len(data['ID'].unique()):
    print("fuck")

data = data.sort_values(by='ID')

data.to_csv("submit.csv", columns=["ID", 'real review?'], index=False)