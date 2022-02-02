# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

# import and combine data
ETFs = pd.read_csv('Morningstar - European ETFs.csv', parse_dates=['inception_date', 'latest_nav_date'])
Mutuals = pd.read_csv('Morningstar - European Mutual Funds.csv', parse_dates=['inception_date', 'latest_nav_date'])

ETFs['Type_ETF'] = 1
Mutuals['Type_ETF'] = 0

print(ETFs.shape)
print(Mutuals.shape)

df = pd.concat([ETFs, Mutuals])

# data cleaning
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
df = df[df['price_prospective_earnings'] > 1]
df = df[df['price_prospective_earnings'] < 100]
df.reset_index(drop=True, inplace=True)

sns.set()
_ = plt.hist(df['price_prospective_earnings'])
_ = plt.xlabel('number of ETFs')
_ = plt.ylabel('PE ratio')
plt.show()

# preparing data for training
features = ['rating', 'risk_rating', 'performance_rating', 'equity_size_score',
            'long_term_projected_earnings_growth', 'historical_earnings_growth', 'sales_growth', 'cash_flow_growth',
            'book_value_growth', 'roa', 'roe', 'roic', 'ongoing_cost', 'management_fees', 'environmental_score',
            'social_score', 'governance_score', 'sustainability_score', 'fund_size', 'fund_trailing_return_ytd',
            'fund_trailing_return_3years', 'fund_trailing_return_5years', 'fund_trailing_return_10years', 'Type_ETF'
            ]

response = ['price_prospective_earnings']

X = df.copy()[features]
y = df.copy()[response]

for col in X:
    X.loc[:, col] = X[col].fillna(np.mean(X[col]))

print(X.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# fitting model
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('reg', Lasso(alpha=0.2))])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

score = pipeline.score(X_test, y_test)

print(score)

cv_results = cross_val_score(pipeline, X, y, cv=10)
print(cv_results)
print(np.mean(cv_results))

# checking predictions
res = pipeline.predict(X).reshape(-1, 1) - y

df['residuals'] = res
df['res_percent'] = res * 100 / y
df['actual'] = y
df['prediction'] = pipeline.predict(X)
df.sort_values('res_percent', inplace=True, ascending=False)

print(df.shape)
print(score)

# visualization of coefficients
coefficient_values = pd.DataFrame([features, pipeline.named_steps['reg'].coef_], index=['features', 'coefficients'])
coefficient_values = coefficient_values.transpose().set_index('features')
coefficient_values_nonzero = coefficient_values[coefficient_values['coefficients'] != 0]

sns.barplot(x=coefficient_values_nonzero.index, y='coefficients', data=coefficient_values_nonzero)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# using formula to create ranking system based on value
def rank_estimate(data=df, target='price_prospective_earnings'):
    data = data.copy()
    X_func = data.copy()[features]
    y_func = pd.DataFrame(data.copy()[target])
    for cols in X_func:
        X_func.loc[:, cols] = X_func[cols].fillna(np.mean(X_func[cols]))
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('reg', Lasso(alpha=0.2))])
    pipe.fit(X_func, y_func)
    data[target + '_percent_residual'] = (pipe.predict(X_func).reshape(-1, 1) - y_func) / y_func
    data.set_index('isin', inplace=True)
    rank = pd.DataFrame(data[target + '_percent_residual'].rank())
    print(pipe.score(X_func, y_func))
    return rank


targets = ['price_prospective_earnings', 'price_book_ratio', 'price_sales_ratio', 'price_cash_flow_ratio',
           'dividend_yield_factor']

ranks = rank_estimate(data=df, target=targets[0])

for t in targets[1:]:
    a = rank_estimate(df, t)
    ranks = ranks.merge(a, on='isin')

mapping = {'price_prospective_earnings_percent_residual': 'a',
           'price_book_ratio_percent_residual': 'b', 'price_sales_ratio_percent_residual': 'c',
           'price_cash_flow_ratio_percent_residual': 'd',
           'dividend_yield_factor_percent_residual': 'e'}

ranks.rename(columns=mapping, inplace=True)
ranks['value_score'] = ranks['a'] * 0.5 + ranks['b'] * 0.125 + ranks['c'] * 0.125 + ranks['d'] * 0.125 \
                       + ranks['e'] * 0.125
ranks.sort_values('value_score', ascending=False, inplace=True)
