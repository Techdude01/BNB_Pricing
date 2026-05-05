import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load data
df = pd.read_csv("AB_NYC_2019.csv")

# Initial Exploration
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())
# Check for missing values
print(df.isnull().sum())
# Imputation of missing values
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df["last_review"] = df["last_review"].fillna("No reviews")
# Remove rows with missing values in 'name' and 'host_name'
df = df.dropna(subset=["name", "host_name"])
# Checking for $0 price listings and removing $0 price listings
print(df[df["price"] == 0].T)
df = df[df["price"] > 0]
print(df["price"].describe())
# Cleaning price outliers with IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
print(upper)
df = df[df["price"] <= upper]
# Confirm cleaning
print(df.shape)
print(df.isnull().sum())
# EDA Visualization
plt.hist(df["price"], bins=50)
plt.title("Distribution of Price")
plt.xlabel("Price ($)")
plt.ylabel("Number of Listings")
plt.show()
# EDA on categorical variables
print(df["room_type"].value_counts())
print(df.groupby("room_type")["price"].mean())
df.groupby("room_type")["price"].mean().plot(kind="bar")
plt.title("Average Price by Room Type")
plt.xlabel("Room Type")
plt.ylabel("Average Price ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.boxplot(column="price", by="room_type")
plt.title("Price Distribution by Room Type")
plt.suptitle("")
plt.xlabel("Room Type")
plt.ylabel("Price ($)")
plt.show()

print(df["neighbourhood_group"].value_counts())
print(df.groupby("neighbourhood_group")["price"].mean())
df.groupby("neighbourhood_group")["price"].mean().plot(kind="bar")
plt.title("Average Price by Neighbourhood Group")
plt.xlabel("Neighbourhood Group")
plt.ylabel("Average Price ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# EDA on numeric variables
plt.hist(df["minimum_nights"], bins=50)
plt.title("Distribution of Minimum Nights")
plt.xlabel("Minimum Nights")
plt.ylabel("Number of Listings")
plt.show()

plt.hist(df["number_of_reviews"], bins=50)
plt.title("Distribution of Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Listings")
plt.show()

plt.hist(df["availability_365"], bins=50)
plt.title("Distribution of Availability (Days per Year)")
plt.xlabel("Available Days per Year")
plt.ylabel("Number of Listings")
plt.show()

print(df[["price", "minimum_nights", "number_of_reviews", "availability_365"]].corr())

# Location Visualization
plt.scatter(df["longitude"], df["latitude"], c=df["price"], cmap="viridis", alpha=0.5)
plt.title("Price by Location in NYC")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label="Price ($)")
plt.show()

# Correlation heatmap
import seaborn as sns

sns.heatmap(
    df[["price", "minimum_nights", "number_of_reviews", "availability_365"]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Heatmap")
plt.show()


# Room Type Distribution by Borough
plt.figure(figsize=(12, 7))
sns.countplot(data=df, x='neighbourhood_group', hue='room_type', palette='viridis')
plt.title('Room Type Distribution by Borough')
plt.xlabel('Borough (Neighbourhood Group)')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Median Price by Borough
median_price_by_borough = df.groupby('neighbourhood_group')['price'].median().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=median_price_by_borough, x='neighbourhood_group', y='price', palette='magma')
plt.title('Median Price by Borough')
plt.xlabel('Borough (Neighbourhood Group)')
plt.ylabel('Median Price ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price Distribution is right-skewed. To make it more symmetrical I'm applying a log transformation. 
df['price_log'] = np.log1p(df['price'])

plt.figure(figsize=(10, 6))
sns.histplot(df['price_log'], kde=True, bins=50)
plt.title('Distribution of Log-Transformed Price')
plt.xlabel('Log(Price)')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.show()

print('--- One-Hot Encoding Categorical Features ---')
# One-hot encode 'room_type'
df_encoded = pd.get_dummies(df, columns=['room_type'], prefix='room')

# One-hot encode 'neighbourhood_group'
df_encoded = pd.get_dummies(df_encoded, columns=['neighbourhood_group'], prefix='borough')

display(df_encoded.head())

print('--- Feature Engineering: Days Since Last Review ---')
# Convert 'last_review' to datetime objects
df_encoded['last_review_date'] = pd.to_datetime(df_encoded['last_review'], errors='coerce')


max_review_date = df_encoded['last_review_date'].max()

df_encoded['days_since_last_review'] = (max_review_date - df_encoded['last_review_date']).dt.days

df_encoded['days_since_last_review'] = df_encoded['days_since_last_review'].fillna(9999)

display(df_encoded[['last_review', 'last_review_date', 'days_since_last_review']].head())

import seaborn as sns

# Select numerical and newly engineered/encoded features for the heatmap
correlation_features = [
    'price_log',
    'minimum_nights',
    'number_of_reviews',
    'calculated_host_listings_count',
    'availability_365',
    'days_since_last_review',
    'room_Entire home/apt',
    'room_Private room',
    'room_Shared room',
    'borough_Bronx',
    'borough_Brooklyn',
    'borough_Manhattan',
    'borough_Queens',
    'borough_Staten Island'
]

# Ensure all selected features exist in df_encoded
existing_features = [f for f in correlation_features if f in df_encoded.columns]

plt.figure(figsize=(14, 10))
sns.heatmap(
    df_encoded[existing_features].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f' # Format annotations to two decimal places
)
plt.title('Updated Correlation Heatmap with Engineered and Encoded Features')
plt.show()