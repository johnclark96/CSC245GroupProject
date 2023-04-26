# BEGINNING OF IMPORTS
import numpy as np  # linear algebra
import pandas as pd
import xgboost as xgb #xgboost should be importing for something? get rid of it if it isn't?
import matplotlib.pyplot as plt
import warnings
import datetime

# END OF IMPORTS

# NOTES:
# df is short for DataFrames (Data is stored inside Dataframes)

warnings.filterwarnings('ignore')

# TODO: Understand the DataSets provided (What does Application Record consist of? What is Credit Record and how do you read it?)
df_application = pd.read_csv('data/application_record.csv')  # df_application reads the application_record excel sheet.
df_credit = pd.read_csv('data/credit_record.csv') # Applicant Credit Records

# BEGINNING OF INITIAL DATA READ

# The Code below outputs what the datas "shape" or how the data is seen.
# Shape will show how much data and how many columns are present.
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

# END OF INITIAL DATA READ

# START OF DATA
# TODO: What is Sample doing here?
df_application.sample(5)
df_credit.sample(5)


# TODO: what is on and how?
# on='ID' appears to be taking the ID columns?
# how='inner' TODO: Someone tell me what that means? (Hint: probably could mouse over pd.merge to see what it does)
df_final = pd.merge(df_application, df_credit, on='ID', how='inner') # merging the 2 data together?

df_final.shape # Shapes the data

df_final.columns

df_final.describe().T


# ------------------------------- ANYTHING BELOW THIS LINE I DID NOT REALLY COMMENT --------------------
# Will go through this later

# Earliest Month
credit_card_first_month = df_final.groupby(['ID']).agg(
    start_month=('MONTHS_BALANCE', min)
).reset_index()
credit_card_first_month.head()

credit_card_first_month['account_open_month'] = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
credit_card_first_month['account_open_month'] = credit_card_first_month['account_open_month'] + credit_card_first_month[
    'start_month'].values.astype("timedelta64[M]")
credit_card_first_month['account_open_month'] = credit_card_first_month['account_open_month'].dt.strftime('%b-%Y')

credit_card_first_month.head()

# join the table
credit_start_status = pd.merge(credit_card_first_month, df_credit, how='left', on=['ID'])

credit_start_status['start_month'] = abs(credit_start_status['start_month']) + credit_start_status['MONTHS_BALANCE']

credit_start_status.head()

credit_start_status['STATUS'].value_counts()

#FIGURE-1-GRAPH
accounts_counts =pd.DataFrame({'start_month':credit_start_status.groupby('start_month')['start_month'].count()})
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(accounts_counts.index,accounts_counts['start_month'])
plt.show()

month_status_counts = credit_start_status.groupby(['start_month', 'STATUS']).size().reset_index(name='counts')
month_counts = credit_start_status.groupby(['start_month']).size().reset_index(name='month_counts')
# join the table
month_status_pct = pd.merge(month_status_counts, month_counts, how='left', on=['start_month'])
month_status_pct['status_pct'] = month_status_pct['counts'] / month_status_pct['month_counts'] * 100
month_status_pct = month_status_pct.loc[:, ['start_month', 'STATUS', 'status_pct']]

# Restructure
month_status_pct1 = month_status_pct.pivot(index='start_month', columns='STATUS', values='status_pct')
# Fill with 0
month_status_pct1 = month_status_pct1.fillna(0).reset_index()

#GRAPH-PLOT
plt.plot(month_status_pct1.index, month_status_pct1['4'] + month_status_pct1['5'],
        color='green',
        linestyle='solid',
        linewidth=2,
        markersize=12)
plt.xlabel('Months Since Opened')
plt.ylabel('% Bad Rate')

month_status_pct2 = month_status_pct1.loc[month_status_pct1.index <= 50]
# drop column start_month
month_status_pct2 = month_status_pct2.drop('start_month', axis=1)

#GRAPH-PLOT
month_status_pct2.plot.area(stacked=True);
plt.show(block=True);

#GRAPH-PLOT
import matplotlib.pyplot as pt

plt.plot(month_status_pct2.index, month_status_pct2['4'] + month_status_pct2['5'],
        color='green',
        linestyle='solid',
        linewidth=2,
        markersize=12)

df_credit['STATUS'].value_counts()

credit_start_status.groupby('STATUS')['STATUS'].count()

credit_start_status1 = credit_start_status.loc[(df_credit['STATUS'] != 'X') & (df_credit['STATUS'] != 'C'), :]

credit_start_status1['status'] = credit_start_status1['STATUS']

credit_start_status1 = credit_start_status1.loc[
    credit_start_status1['start_month'] <= 18, ['ID', 'start_month', 'status']]

credit_start_status1 = credit_start_status1[(credit_start_status1['status'] != 'C')]

credit_start_status1 = credit_start_status1[(credit_start_status1['status'] != 'X')]

credit_start_status1

# Find Max Status Values
status = credit_start_status1.groupby(['ID']).agg(
    # Max Status
    max_status=('status', 'max')

).reset_index()
# Validate
status.groupby('max_status')['max_status'].count()

# Define
status['label'] = np.where(status['max_status'].astype(int) >= int(4), 1, 0)
# Validate
status.groupby('label')['label'].count()

status.groupby('label')['label'].count() * 100 / len(status['label'])

# All with label 1
label_1 = status.loc[status['label'] == 1, :]
# All with label 0
label_0 = status.loc[status['label'] == 0, :]
# Select randomly few rows
label_0_biased = label_0.sample(n=1701)
# Combined Sample IDs with Biased Sampling

frames = [label_1, label_0_biased]
labels_biased = pd.concat(frames)

# Keep only ID and Label Columns

labels_biased = labels_biased.loc[:, ['ID', 'label']]

labels_biased

# Combine Labels and Application Data
model_df = pd.merge(labels_biased, df_application, how='inner', on=['ID'])
len(model_df)

model_df.tail()

model_df.groupby('label')['label'].count() * 100 / len(model_df['label'])


# Check if missing values
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


# source: https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe

missing_values_table(model_df)


# Find Continuous and Categorical Features
def featureType(df):
    import numpy as np
    from pandas.api.types import is_numeric_dtype

    columns = df.columns
    rows = len(df)
    colTypeBase = []
    colType = []
    for col in columns:
        try:
            try:
                uniq = len(np.unique(df[col]))
            except:
                uniq = len(df.groupby(col)[col].count())
            if rows > 10:
                if is_numeric_dtype(df[col]):
                    if uniq == 1:
                        colType.append('Unary')
                        colTypeBase.append('Unary')
                    elif uniq == 2:
                        colType.append('Binary')
                        colTypeBase.append('Binary')
                    elif rows / uniq > 3 and uniq > 5:
                        colType.append('Continuous')
                        colTypeBase.append('Continuous')
                    else:
                        colType.append('Continuous-Ordinal')
                        colTypeBase.append('Ordinal')
                else:
                    if uniq == 1:
                        colType.append('Unary')
                        colTypeBase.append('Category-Unary')
                    elif uniq == 2:
                        colType.append('Binary')
                        colTypeBase.append('Category-Binary')
                    else:
                        colType.append('Categorical-Nominal')
                        colTypeBase.append('Nominal')
            else:
                if is_numeric_dtype(df[col]):
                    colType.append('Numeric')
                    colTypeBase.append('Numeric')
                else:
                    colType.append('Non-numeric')
                    colTypeBase.append('Non-numeric')
        except:
            colType.append('Issue')

        # Create dataframe
    df_out = pd.DataFrame({'Feature': columns,
                           'BaseFeatureType': colTypeBase,
                           'AnalysisFeatureType': colType})
    return df_out


featureType(model_df)

from datetime import timedelta

model_df['BIRTH_DATE'] = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d") + model_df['DAYS_BIRTH'].apply(
    pd.offsets.Day)

# DAYS_EMPLOYED: Count backwards from current day(0). If positive, it means the person currently unemployed.
# Update DAYS_EMPLOYED greater than 0 to 31
model_df.loc[model_df.DAYS_EMPLOYED > 0, "DAYS_EMPLOYED"] = 31
model_df['EMPLOYMENT_START_DATE'] = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d") + model_df[
    'DAYS_EMPLOYED'].apply(pd.offsets.Day)

model_df.head()

model_df = pd.merge(model_df, credit_card_first_month.loc[:, ['ID', 'account_open_month']], how='inner', on=['ID'])
len(model_df)

# Age in months

model_df['age_months'] = (
            (pd.to_datetime(model_df['account_open_month'], format='%b-%Y') - model_df.BIRTH_DATE) / np.timedelta64(1,
                                                                                                                    'M'))
model_df['age_months'] = model_df['age_months'].astype(int)
# Experience/Employment in Months
model_df['employment_months'] = ((pd.to_datetime(model_df['account_open_month'],
                                                 format='%b-%Y') - model_df.EMPLOYMENT_START_DATE) / np.timedelta64(1,
                                                                                                                    'M'))
model_df['employment_months'] = model_df['employment_months'].astype(int)

model_df.loc[model_df.employment_months < 0, "employment_months"] = -1

model_df = model_df.drop(
    ['BIRTH_DATE', 'EMPLOYMENT_START_DATE', 'account_open_month', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL'], axis=1)

featureType(model_df)

import warnings

warnings.filterwarnings("ignore")

income_type = model_df.groupby(['NAME_INCOME_TYPE', 'label'])['NAME_INCOME_TYPE', 'label'].size().reset_index(
    name='counts')

# Restucture
income_type = income_type.pivot(index='NAME_INCOME_TYPE', columns='label', values='counts')
# Fill with 0
income_type = income_type.fillna(0).reset_index()
# Rename the columns
income_type.columns = ['Income_Type', 'Label_0', 'Label_1']

# Calculate Bad Rate for each of the income type
income_type['pct_obs'] = (income_type['Label_0'] + income_type['Label_1']) / (
            sum(income_type['Label_0']) + sum(income_type['Label_1']))
income_type['pct_label_0'] = income_type['Label_0'] / (income_type['Label_0'] + income_type['Label_1'])
income_type['pct_label_1'] = income_type['Label_1'] / (income_type['Label_0'] + income_type['Label_1'])
print(income_type)

# change missing value for OCCUPATION_TYPE
model_df.loc[model_df.OCCUPATION_TYPE == '', "OCCUPATION_TYPE"] = "NA"
# One hot Encoding using get_dummies function
model_df2 = pd.get_dummies(model_df, columns=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', "NAME_INCOME_TYPE",
                                              'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                                              'OCCUPATION_TYPE'])

len(model_df2)

from sklearn.metrics import accuracy_score

model_df2.columns

# Features - exclude ID and Label columns
features = model_df2.iloc[:, 2:]
# Label - select only label column
label = model_df2.iloc[:, 1]

label

model_df2.sample(5)

model_df2.dtypes

from sklearn.model_selection import train_test_split

features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2,
                                                                          random_state=557)

from pycaret.regression import *
from pycaret.classification import *

reg_experiment = setup(model_df2,
                       target='label',
                       session_id=42,
                       experiment_name='credit_card_approval',
                       normalize=True,
                       transformation=True,
                       remove_multicollinearity=True,
                       # rop one of the two features that are highly correlated with each other
                       multicollinearity_threshold=0.5
                       )

best_model = compare_models()

rf = create_model('rf')

rf = tune_model(rf, optimize='F1')

plot_model(rf)

plot_model(rf, plot='feature')

print(evaluate_model(rf))

interpret_model(rf)

pred_holdouts = predict_model(rf)
pred_holdouts.head()

save_model(tuned_catboost, model_name='./random_forest')
