# -*- coding: utf-8 -*-
"""**Import Library yang dibutuhkan**"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

"""**Import Dataset & Pemisahan Atribut dan Label**"""

# Import Data
dataset = pd.read_csv('Stroke_Data.csv')
# Pemisahan Atribut dan Label
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[: , -1].values

"""Matrix x yang berisi Atribut"""

print(x)

"""Matrix y yang berisi Label (kolom Stroke)"""

print(y)

"""Menghilangkan Missing Value Menggunakan Strategy **Mean**"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Kolom ke-9 , index ke-8 (kolom bmi terdapat N/A)
imputer.fit(x[:, [8]])
x[:, [8]] = imputer.transform(x[:, [8]])

print(x)

"""Menghilangkan Missing Value Menggunakan Strategy **Median**"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# Kolom ke-9 , index ke-8 (kolom bmi terdapat N/A)
imputer.fit(x[:, [8]])
x[:, [8]] = imputer.transform(x[:, [8]])

print(x)

"""Menghilangkan Missing Value Menggunakan Strategy **Most_Frequent**"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# Kolom ke-10 , index ke-9 (kolom bmi terdapat N/A)
imputer.fit(x[:, [8]])
x[:, [8]] = imputer.transform(x[:, [8]])

print(x)

"""Encoding Data Kategori (Atribut)"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

"""Data Kategori (Class / Label) tidak perlu encoding karena sudah berupa numeric (0,1)

Membagi Dataset kedalam training set  dan test set
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(x_train)

print(x_test)

print(y_train)

print(y_test)