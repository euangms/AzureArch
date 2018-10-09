import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

df = pd.read_csv('C:\\dev\Demos\\AzureArch\\Exercise\\dirty_titanic.csv', sep = '|', skiprows = 4, skipfooter = 1, engine = 'python')

###Magic happens here
df = df.drop(['home.dest', 'ticket', 'name', 'boat', 'body', 'cabin'],axis=1)

df = map_sex(df)
df = impute_age_median(df)
df2 = df.dropna()
df2 = scale_fares(df2)
df2 = handle_embarked(df2)
df2.drop('embarked', axis=1, inplace=True)

df2.to_csv('C:\\dev\Demos\\AzureArch\\Exercise\\clean_titanic.csv', index = False)
