# include module
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# the function of data cleaning 
def data_cleaning(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # 將 training data 與 testing data 合併
    total = pd.concat([train, test], axis = 0)

    # some features which will drop nan
    drop_list = ['PoolQC', 'Fence']
    total.drop(columns = drop_list, inplace = True)

    # some features which will fill with 'Na'
    fill_list = ['MiscFeature', 'Alley', 'FireplaceQu', 'GarageFinish', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
    for feature in fill_list:
        total[feature] = total[feature].fillna('Na')

    # some features which will fill with mean
    mean_list = ['GarageYrBlt']
    for feature in mean_list:
        total[feature] = total[feature].fillna(train[feature].mean())
    
    le = LabelEncoder()
    imputer = KNNImputer(n_neighbors = 1)
    # knn imputer
    sub1 = total.loc[:,['LotFrontage', 'Street', 'Alley']]
    sub1['Street'] = le.fit_transform(sub1['Street'])
    sub1['Alley'] = le.fit_transform(sub1['Alley'])
    sub1 = pd.DataFrame(imputer.fit_transform(sub1))
    feature = 'LotFrontage'
    total[feature] = total[feature].fillna(sub1.iloc[:,0])

    sub2 = total.loc[:,['MasVnrType','MasVnrArea']]
    sub2['MasVnrType'] = le.fit_transform(sub2['MasVnrType'])
    sub2 = pd.DataFrame(imputer.fit_transform(sub2))
    feature = 'MasVnrArea'
    total[feature] = total[feature].fillna(sub2.iloc[:,1])

    # GarageType 眾數
    feature = 'GarageType'
    mode = train[(train[feature] != 'BuiltIn') & (train[feature] != 'Attchd') & (train[feature] != 'Detchd')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # GarageQual 眾數
    feature = 'GarageQual'
    mode = train[(train[feature] != 'TA')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # GarageCond 眾數
    feature = 'GarageCond'
    mode = train[(train[feature] != 'TA')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # BsmtFinType2 眾數
    feature = 'BsmtFinType2'
    mode = train[(train[feature] != 'Unf')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # MasVnrType
    feature = 'MasVnrType'
    total[feature] = total[feature].fillna('BrkCmn')

    # Electrical
    feature = 'Electrical'
    total[feature] = total[feature].fillna('FuseF')

    return total