import pandas as pd

df = pd.read_csv('C:\\dev\Demos\\AzureArch\\Exercise\\dirty_titanic.csv', sep = '|', skiprows = 4, skipfooter = 1, engine = 'python')


'''
Magic happens here
'''


df.to_csv('C:\\dev\Demos\\AzureArch\\Exercise\\clean_titanic.csv', index = False)
