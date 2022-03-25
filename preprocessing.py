# impor library
import numpy as np
import pandas as pd

data = {
    'Province': ['Banten', 'DKI Jakarta','Jawa Barat','Banten','Jawa Barat','DKI Jakarta','Banten','Banten','Jawa Barat','DKI Jakarta','Banten','Banten','Jawa Barat','DKI Jakarta','DKI Jakarta'],
    'Age': [24,np.nan,60,34,58,np.nan,21,44,40,51,32,30,30,19,25],
    'Wage': [5000000,3400000,7350000,3500000,np.nan,8000000,5500000,10000000,9000000,10500000,np.nan,6400000,np.nan,2200000,4500000],
    'Life insured': ['Yes','No','No','No','Yes','No','No','Yes','Yes','Yes','No','No','No','Yes','Yes'],
}

# impor dataset
dataset = pd.read_csv(
        'contoh_dataset.csv',
        delimiter=';', 
        header='infer', 
        index_col=False
        )

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# menangani nilai kosong
from sklearn.impute import SimpleImputer

# ganti NaN dengan mean kolom itu
imputer = SimpleImputer(
        missing_values=np.nan, 
        strategy='mean'
        )
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# kodekan data kategori
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# kodekan nama provinsi (kolom ke-0)
# kode hanya sebatas penanda
encoder_X = ColumnTransformer(
        [('province_encoder', OneHotEncoder(), [0])], 
        remainder='passthrough'
        )
X = encoder_X.fit_transform(X).astype(float) # mengembalikan ke dalam tipe 'float64'

print(encoder_X.named_transformers_['country_encoder'].categories_)

# y adalah dependent, cukup kodekan ke angka
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# kemudian:
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# variabel dummy kode provinsi juga diskalakan
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(-1, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)