# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk(''):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np  # linear algebra
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df_application = pd.read_csv('data/application_record.csv')
df_credit = pd.read_csv('data/credit_record.csv')

print("Shape of application data", df_application.shape)
print("-------------------------------------------")
print("Shape of credit data", df_credit.shape)

print("Columns of application data", df_application.columns)
print("-------------------------------------------")
print("Columns of credit data", df_credit.columns)

print("data type of application data", df_application.dtypes)
print("-------------------------------------------")
print("data type of credit data", df_credit.dtypes)

print("columns in application data")
print("--------------------------------------------------------------------------------")
cat_app_data = [i for i in df_application.select_dtypes(include=np.object).columns]
num_app_data = [i for i in df_application.select_dtypes(include=np.number).columns]
print("categorical columns in application data", cat_app_data)
print("--------------------------------------------------------------------------------")
print("numerical columns in application data", num_app_data)

print("columns in credit data")
print("--------------------------------------------------------------------------------")
cat_credit_data = [i for i in df_credit.select_dtypes(include=np.object).columns]
num_credit_data = [i for i in df_credit.select_dtypes(include=np.number).columns]
print("categorical columns in application data", cat_credit_data)
print("--------------------------------------------------------------------------------")
print("numerical columns in application data", num_credit_data)

df_application.sample(5)


df_credit.sample(5)

df_final = pd.merge(df_application, df_credit, on='ID', how='inner')


df_final.shape

df_final.columns

df_final.describe().T


# Earliest Month
credit_card_first_month =df_final.groupby(['ID']).agg(
   start_month=  ('MONTHS_BALANCE', min)
    ).reset_index()
credit_card_first_month.head()

import datetime
credit_card_first_month['account_open_month']= datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
credit_card_first_month['account_open_month']= credit_card_first_month['account_open_month'] + credit_card_first_month['start_month'].values.astype("timedelta64[M]")
credit_card_first_month['account_open_month']=credit_card_first_month['account_open_month'].dt.strftime('%b-%Y')

