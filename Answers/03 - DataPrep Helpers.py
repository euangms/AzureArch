def impute_age_median(source_df):
    MedianAge = source_df['age'].median()
    source_df['age'] = source_df['age'].fillna(value=MedianAge)
    return source_df
    
def scale_fares(source_df):
    scale = StandardScaler().fit(source_df[['fare']])
    source_df[['fare']] = scale.transform(source_df[['fare']])    
    return source_df

def dummy_embarked(source_df):
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    source_df['embarked'].fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(source_df['embarked'], prefix='embarked')
    source_df = pd.concat([source_df, embarked_dummies], axis=1)
    return source_df

def handle_embarked(source_df):
    ModeEmbarked = source_df['embarked'].mode()[0]
    source_df['embarked'] = source_df['embarked'].fillna(value=ModeEmbarked)
    embarked_dummies = pd.get_dummies(source_df['embarked'], prefix='embarked')
    source_df = pd.concat([source_df, embarked_dummies], axis=1)
    return source_df

def map_sex(source_df):
    # mapping string values to numerical one 
    source_df['sex'] = source_df['sex'].map({'male':1, 'female':0})
    return source_df


