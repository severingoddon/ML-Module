import pandas as pd
import re
from matplotlib import pyplot as plt

df = pd.read_csv('sales.csv')

# get a first overview
print(df.head())
print(df.shape)
print(df.dtypes)

#change column datetype of 'ordered_at' to datetime type
df['ordered_at'] = pd.to_datetime(df['ordered_at'])

#remove $ signs and change datatype to float at both columns price and line_total
for column in ['price', 'line_total']:
    df[column] = df[column].apply(lambda x: float(x[1:]))

# show amount of duplicates
print(df[df.duplicated()].shape[0])
# remove duplicates
df = df.drop_duplicates()
# check if order_id is unique?
print(df['order_id'].is_unique) # false --> is not unique

# check if there are null values
print(df.isnull().sum()) # 1481 values in column name are null
print(df[df['name'].isnull()].head())

#remove NaN values
df = df.dropna()

# check for wrong calculations
print(df[(df['price'] * df['quantity']) != df['line_total']].shape[0])

# remove the wrong ones
df = df[(df['price'] * df['quantity']) == df['line_total']]

print(df.describe())
# check for negative line_total entries
print(df[df['line_total'] < 0].shape[0])

# remove negative line_total entries
df = df[df['line_total'] >= 0]

# define regex pattern
pattern = r'^"([A-Z ]+)" (.*)'
transform_func = lambda x: re.findall(pattern, x)[0]

# add a new column "category" and store the category values in it
df[['category', 'name']] = df['name'].apply(transform_func).apply(pd.Series)

print(df.head())

# do some graphical analysis

# check the line_total of each name (like Dark chocolate)
f, ax = plt.subplots(figsize=(10, 6))
df.groupby('name')['line_total'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
f.autofmt_xdate()
plt.show()

# check the line_total of each category (like Sorbet or Ice Cream)
f, ax = plt.subplots(figsize=(8, 6))
df.groupby('category')['line_total'].sum().sort_values(ascending=False).plot(kind='bar')
f.autofmt_xdate()
plt.show()

# show orders during the year
f, ax = plt.subplots(figsize=(17, 6))
df.resample('W', on='ordered_at')['line_total'].sum().plot()
plt.show()