# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import regex as re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import investpy

# Import and combine data

ETFs = pd.read_csv('Morningstar - European ETFs.csv', parse_dates=['inception_date', 'latest_nav_date'])
Mutuals = pd.read_csv('Morningstar - European Mutual Funds.csv', parse_dates=['inception_date', 'latest_nav_date'])

ETFs['Type_ETF'] = 1
Mutuals['Type_ETF'] = 0

print(ETFs.shape)
print(Mutuals.shape)

df = pd.concat([ETFs, Mutuals])

# Data cleaning

df.drop_duplicates(subset=['isin'], inplace=True)
df.drop_duplicates(subset=['price_book_ratio', 'price_sales_ratio', 'price_cash_flow_ratio', 'dividend_yield_factor'],
                   inplace=True)

print(ETFs.info())
print(ETFs.head())
print(ETFs.columns)
print(df.shape)
print(df.head())

df['Days_active'] = df['latest_nav_date'] - df['inception_date']
df = df[df['Days_active'] > dt.timedelta(days=5 * 365)]
df = df[df['price_prospective_earnings'].notna()]
df = df[df['price_book_ratio'].notna()]
df = df[df['price_sales_ratio'].notna()]
df = df[df['price_cash_flow_ratio'].notna()]
df = df[df['dividend_yield_factor'].notna()]
df = df[df['asset_stock'] > 90]
df = df[df['price_prospective_earnings'] > 5]
df = df[df['price_prospective_earnings'] < 75]
df.reset_index(drop=True, inplace=True)

# Regex

country_exp = df.country_exposure
country_list = []

for row in range(len(df)):
    country_percentage = re.findall(r"(\w\w\w)\:\s(\d+)", country_exp[row])
    for country, percentage in country_percentage:
        if int(percentage) < 5:
            break
        elif country in df.columns:
            df.loc[row, country] = int(percentage)
        else:
            country_list.append(country)
            df[country] = 0
            df.loc[row, country] = int(percentage)

# Simple histogram of PE ratios

sns.set()
_ = plt.hist(df['price_prospective_earnings'], bins=20)
_ = plt.xlabel('PE ratio')
_ = plt.ylabel('Number of funds')
plt.savefig('PE ratios.png')

print(f"The average PE ratio in the dataframe is {np.mean(df['price_prospective_earnings']):.2f}")

# Preparing data for training

features = ['rating', 'risk_rating', 'performance_rating', 'equity_size_score', 'long_term_projected_earnings_growth',
            'historical_earnings_growth', 'sales_growth', 'cash_flow_growth', 'book_value_growth', 'roa', 'roe', 'roic',
            'ongoing_cost', 'management_fees', 'environmental_score', 'social_score', 'governance_score',
            'sustainability_score', 'fund_size', 'fund_trailing_return_ytd', 'fund_trailing_return_3years',
            'fund_trailing_return_5years', 'fund_trailing_return_10years', 'Type_ETF']

for country in country_list:
    features.append(country)

response = ['price_prospective_earnings']

X = df.copy()[features]
y = df.copy()[response]

for col in X:
    X.loc[:, col] = X[col].fillna(np.mean(X[col]))

print(X.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# Fitting first regression model

pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('reg', Lasso(alpha=0.5))])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

score = pipeline.score(X_test, y_test)

cv_results = cross_val_score(pipeline, X, y, cv=10)

print(f"The average score obtained was {np.mean(cv_results)*100:.2f} %")

# Checking predictions

residuals = pipeline.predict(X).reshape(-1, 1) - y

df['residuals_percent'] = residuals * 100 / y
df['actual_value'] = y
df['prediction'] = pipeline.predict(X)

reg_predictions = df.sort_values('residuals_percent', ascending=False)

print("The best funds based on this regression are shown below:")
print(reg_predictions.head())

# Visualization of coefficients

coefficient_values = pd.DataFrame([features, pipeline.named_steps['reg'].coef_], index=['features', 'coefficients'])
coefficient_values = coefficient_values.transpose().set_index('features')
coefficient_values_nonzero = coefficient_values[coefficient_values['coefficients'] != 0]

sns.barplot(x=coefficient_values_nonzero.index, y='coefficients', data=coefficient_values_nonzero)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('Principle Component Analysis.png')


# Using formula to create ranking system based on value

def rank_estimate(data=df, target='price_prospective_earnings', reg=Lasso(alpha=0.5)):
    data = data.copy()
    X_func = data.copy()[features]
    y_func = pd.DataFrame(data.copy()[target])
    for cols in X_func:
        X_func.loc[:, cols] = X_func[cols].fillna(np.mean(X_func[cols]))
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('reg', reg)])
    pipe.fit(X_func, y_func)
    data[target + '_percent_residual'] = (pipe.predict(X_func).reshape(-1, 1) - y_func) / y_func
    data.set_index('isin', inplace=True)
    rank = pd.DataFrame(data[target + '_percent_residual'].rank())
    print(f"The accuracy score for {target} is {pipe.score(X_func, y_func)*100:.2f} %")
    return rank


targets = ['price_prospective_earnings', 'price_book_ratio', 'price_sales_ratio', 'price_cash_flow_ratio',
           'dividend_yield_factor']

ranks = rank_estimate(data=df, target=targets[0])

for t in targets[1:]:
    a = rank_estimate(df, t)
    ranks = ranks.merge(a, on='isin')

mapping = {'price_prospective_earnings_percent_residual': 'P/E',
           'price_book_ratio_percent_residual': 'P/B', 'price_sales_ratio_percent_residual': 'P/S',
           'price_cash_flow_ratio_percent_residual': 'P/CF',
           'dividend_yield_factor_percent_residual': 'Div'}

ranks.rename(columns=mapping, inplace=True)
ranks['value_score'] = ranks['P/E'] * 0.5 + ranks['P/B'] * 0.125 + ranks['P/S'] * 0.125 + ranks['P/CF'] * 0.125 \
                       + ranks['Div'] * 0.125
ranks.sort_values('value_score', ascending=False, inplace=True)

print("\nThe best funds based on the this ranking system are given below")
print(ranks.head())

# Showing heatmap of correlation between value factors

sns.heatmap(df[targets].corr(), linewidths=.5, annot=True)
plt.xticks(rotation=45)
plt.savefig('Heatmap of value metrics.png')

# Boosting and improving the model to predict PE ratio only

targets = targets[:-1]

features.append(['dividend_yield_factor_percent_residual', 'price_book_ratio', 'price_sales_ratio',
                 'price_cash_flow_ratio'])

X['bvg2'] = X['book_value_growth'] ** 2
X['bvg3'] = X['book_value_growth'] ** 3
X['ss2'] = X['sustainability_score'] ** 2
X['ss3'] = X['sustainability_score'] ** 3
X['ret1_2'] = X['fund_trailing_return_ytd'] ** 2
X['ret1_3'] = X['fund_trailing_return_ytd'] ** 3
X['ret3_2'] = X['fund_trailing_return_3years'] ** 2
X['ret3_3'] = X['fund_trailing_return_3years'] ** 3
X['ret5_2'] = X['fund_trailing_return_5years'] ** 2
X['ret5_3'] = X['fund_trailing_return_5years'] ** 3
X['ret10_2'] = X['fund_trailing_return_10years'] ** 2
X['ret10_3'] = X['fund_trailing_return_10years'] ** 3

features.append(['bvg2', 'bvg3', 'ss2', 'ss3', 'ret1_2', 'ret1_3', 'ret3_2', 'ret3_3', 'ret5_2', 'ret5_3', 'ret10_2',
                 'ret10_3'])

# Applying a deep learning gradient decent model

scalerX = StandardScaler()
scalerX.fit(X)
standardizedX = scalerX.transform(X)
predictors = standardizedX

scaler_y = StandardScaler()
scaler_y.fit(y)
standardized_y = scaler_y.transform(y)
target_K = standardized_y

X_train_gd, X_test_gd, y_train_gd, y_test_gd = train_test_split(predictors, target_K, test_size=0.20, random_state=123)
n_cols = X_train_gd.shape[1]

model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(n_cols,)))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_gd, y_train_gd, epochs=100, validation_split=0.20, batch_size=250, shuffle=True)

print(f"\nThe revised accuracy using a GD model is {r2_score(y_test_gd, model.predict(X_test_gd))*100:.2f} %!")

model.save('model_file.h5')

my_model = load_model('model_file.h5')
my_model.summary()

# Rescaling the data for use in analysis

new_predictions = model.predict(predictors)

prediction_list = scaler_y.inverse_transform(new_predictions)

df['new_predictions'] = prediction_list.reshape(-1, ).tolist()

df.to_csv('funds_with_predictions.csv')

df['new_residuals'] = (df['new_predictions'] - df['actual_value']) / df['actual_value']
sample = df.sort_values('new_residuals', ascending=False).head()
print(sample['isin'].head())

# Using investpy to reference fund ISINs in lookup - example

etf = search_results = investpy.search_quotes(text='LU0119620416', products=['funds'])
etf = etf[0]
etf_info = etf.retrieve_information()
print("\n")
print(etf)
print(etf_info)

