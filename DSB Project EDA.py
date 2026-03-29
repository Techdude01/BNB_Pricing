import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# load data
df = pd.read_csv("AB_NYC_2019.csv")

# Initial Exploration 
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())
# Check for missing values
print(df.isnull().sum())
#Imputation of missing values
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['last_review'] = df['last_review'].fillna('No reviews')
#Remove rows with missing values in 'name' and 'host_name'
df = df.dropna(subset=['name', 'host_name'])
#Checking for $0 price listings and removing $0 price listings
print(df[df['price'] == 0].T)
df = df[df['price'] > 0]
print(df['price'].describe())
#Cleaning price outliers with IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
print(upper)
df = df[df['price'] <= upper]
#Confirm cleaning
print(df.shape)
print(df.isnull().sum())
#EDA Visualization
plt.hist(df['price'], bins=50)
plt.title('Distribution of Price')
plt.xlabel('Price ($)')
plt.ylabel('Number of Listings')
plt.show()
#EDA on categorical variables
print(df['room_type'].value_counts())
print(df.groupby('room_type')['price'].mean())
df.groupby('room_type')['price'].mean().plot(kind='bar')
plt.title('Average Price by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.boxplot(column='price', by='room_type')
plt.title('Price Distribution by Room Type')
plt.suptitle('')
plt.xlabel('Room Type')
plt.ylabel('Price ($)')
plt.show()

print(df['neighbourhood_group'].value_counts())
print(df.groupby('neighbourhood_group')['price'].mean())
df.groupby('neighbourhood_group')['price'].mean().plot(kind='bar')
plt.title('Average Price by Neighbourhood Group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#EDA on numeric variables
plt.hist(df['minimum_nights'], bins=50)
plt.title('Distribution of Minimum Nights')
plt.xlabel('Minimum Nights')
plt.ylabel('Number of Listings')
plt.show()

plt.hist(df['number_of_reviews'], bins=50)
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Listings')
plt.show()

plt.hist(df['availability_365'], bins=50)
plt.title('Distribution of Availability (Days per Year)')
plt.xlabel('Available Days per Year')
plt.ylabel('Number of Listings')
plt.show()

print(df[['price', 'minimum_nights', 'number_of_reviews', 'availability_365']].corr())

#Location Visualization
plt.scatter(df['longitude'], df['latitude'], c=df['price'], cmap='viridis', alpha=0.5)
plt.title('Price by Location in NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Price ($)')
plt.show()

#Correlation heatmap
import seaborn as sns
sns.heatmap(df[['price', 'minimum_nights', 'number_of_reviews', 'availability_365']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()