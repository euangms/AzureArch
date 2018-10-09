import pandas as pd

df = pd.read_csv('C:\\dev\Demos\\AzureArch\\Exercise\\dirty_titanic.csv', sep = '|', skiprows = 4, skipfooter = 1, engine = 'python')

###Magic happens here
df = df.drop(['embarked','home.dest', 'ticket', 'sex', 'name', 'boat', 'body', 'cabin'],axis=1)
df2 = df.dropna()

df2.to_csv('C:\\dev\Demos\\AzureArch\\Exercise\\clean_titanic.csv', index = False)
