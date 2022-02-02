# import libraries
import numpy as np
import pandas as pd
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
df = df[df['asset_stock'] > 90]
df = df[df['price_prospective_earnings'] > 1]
df = df[df['price_prospective_earnings'] < 100]

# preparing data for training
features = ['rating', 'risk_rating', 'performance_rating', 'equity_style_score', 'equity_size_score',
            'long_term_projected_earnings_growth', 'historical_earnings_growth', 'sales_growth', 'cash_flow_growth',
            'book_value_growth', 'roa', 'roe', 'roic', 'ongoing_cost', 'management_fees', 'environmental_score',
            'social_score', 'governance_score', 'sustainability_score', 'fund_size', 'fund_trailing_return_ytd',
            'fund_trailing_return_3years', 'fund_trailing_return_5years', 'fund_trailing_return_10years', 'Type_ETF'
            ]

target = ['price_prospective_earnings']

X = df.copy()[features]
y = df.copy()[target]

for col in X:
    X.loc[:, col] = X[col].fillna(np.mean(X[col]))

print(X.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# fitting model
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('reg', Lasso(alpha=0.1))])

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
