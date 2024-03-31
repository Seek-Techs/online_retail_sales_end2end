import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data
@st.cache_data
def load_data():
    data = pickle.load(open('online_retail.pkl', 'rb'))
    return data

data = load_data()


# Title
st.title("Retail Sales Analysis")

# Section 1: Overview
st.header("Overview")
st.write("This dashboard presents the findings from the analysis of retail sales data.")

# Shape of the data
shape = data.shape
st.write(data.describe())
st.write('The dataset contains', shape[0], 'rows and', shape[1], 'features')

st.write('check about data:', data.info())

st.header('Check for null values')
st.write(data.isnull().sum())

# Impute missing values in the 'Description' column with the mode (most frequent value)
data['Description'] = data['Description'].fillna(data['Description'].mode()[0])

# Drop rows with missing CustomerID values
data.dropna(subset=['CustomerID'], inplace=True)

# Filter the dataset to identify discount transactions
discount_transactions = data[data['Description'].str.contains('Discount', case=False)]

st.header("Discount Transactions:")
st.write(discount_transactions)

rows, columns = discount_transactions.shape
st.write(f'Discount transactions contain: {rows} numbers')
# st.write(f'Number of columns: {columns}')


# Calculate revenue from discount transactions
discount_revenue = (discount_transactions['Quantity'] * discount_transactions['UnitPrice']).sum()

# Calculate revenue from non-discount transactions
non_discount_revenue = ((data[data['Quantity'] > 0]['Quantity'] * data[data['Quantity'] > 0]['UnitPrice'])).sum()

discount_impact_on_revenue = non_discount_revenue - discount_revenue

# Compare revenue
st.write("Revenue from Discount Transactions:", round(abs(discount_revenue),2))
st.write("Revenue from Non-Discount Transactions:", round(non_discount_revenue, 2))
st.write('So, the discount impact on revenue is approximately $', round(discount_impact_on_revenue, 2), '. This indicates the difference in revenue between transactions with and without discounts.')

st.header('Quantity Purchased During Discounts')

# Group by discount events and calculate total quantity purchased
discount_quantity = discount_transactions.groupby('Description')['Quantity'].sum()
st.write(discount_quantity)
st.write('Number of quantity purchased during discount:', discount_quantity[0])

st.header('Discount Impact on Revenue')

# Calculate revenue from discount transactions
discount_revenue = (discount_transactions['Quantity'] * discount_transactions['UnitPrice']).sum()

# Calculate revenue from non-discount transactions
non_discount_revenue = ((data[data['Quantity'] > 0]['Quantity'] * data[data['Quantity'] > 0]['UnitPrice'])).sum()

discount_impact_on_revenue = non_discount_revenue - discount_revenue

import matplotlib.pyplot as plt

# Calculate total revenue from discount and non-discount transactions
total_discount_revenue = abs(discount_revenue)
total_non_discount_revenue = non_discount_revenue

# Data for the pie chart
labels = ['Discount Revenue', 'Non-Discount Revenue']
sizes = [total_discount_revenue, total_non_discount_revenue]

# Plot the pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
plt.title('Distribution of Revenue from Discount and Non-Discount Transactions', fontsize=16)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Show the plot
st.pyplot(fig)

# Offering a 10% discount for purchases of 10 or more units
bulk_discount = 0.1 # Assuming a 10% discount for bulk purchase

# calculate and Create a total price field
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
data['TotalPrice'].sum()

# Filter the dataset to include only discount transactions
discount_transactions = data[(data['Description'] == 'Discount') & (data['Quantity'] < 0)]

# Calculate the impact on revenue
total_revenue_impact = abs(discount_transactions['TotalPrice'].sum())

# # Calculate the frequency of discount transactions
# discount_frequency = len(discount_transactions)

# # Calculate the total amount discounted
# total_discount_amount = discount_transactions['TotalPrice'].sum()

# st.write("Frequency of discount transactions:", discount_frequency)
# st.write("Total amount discounted:", round(total_discount_amount, 2))
st.write("Impact on revenue (absolute value): $", round(total_revenue_impact, 2))

# Calculate revenue from discount transactions
discount_revenue = (discount_transactions['Quantity'] * discount_transactions['UnitPrice']).sum()

st.write('The difference in revenue between discount and non-discount transactions is substantial, indicating that discounts have a notable impact on overall revenue.')

st.header('Pricing Strategies to Affect Customer Buying Behaviour')

# Offering a 10% discount for purchases of 10 or more units
bulk_discount = 0.1 # Assuming a 10% discount for bulk purchase

# create a discounted price field
data['DiscountedPrice'] = data['TotalPrice']
data.loc[data['Quantity'] >= 10, 'DiscountedPrice'] *= (1 - bulk_discount)

# Display the updated DataFrame with discounted prices
st.write("\nUpdated DataFrame with Discounted Prices:")
st.write(data.head())

# Section 2: Discount Impact Analysis
description = data['Description'].nunique()
products_categories = data['Description'].value_counts().head(10)
st.header('Discount Impact Analysis on Product')
st.write(products_categories)
st.write('Number of  different categories of the product:', description)

country_categories = data['Country'].nunique()
top_countries = data['Country'].value_counts().head(10)
st.header('Discount Impact Analysis on Country')
st.write(top_countries)

# Plotting the distribution of orders by country
fig, ax = plt.subplots(figsize=(10, 6))
top_countries.plot(kind='bar', color='skyblue')
plt.title('Top 10 Countries by Order Frequency')
plt.xlabel('Country')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
st.pyplot(fig)
st.write('Number of country present:', country_categories)

top_customer = data['CustomerID'].nunique()
top_customer = data['CustomerID'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 6))
top_customer.plot(kind='bar', color='skyblue')
plt.title('Top 10 Customers by Order Frequency')
plt.xlabel('Customer')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
st.pyplot(fig)
st.write('Number of Customer present:', top_customer)
st.header('Average Unit Price of each products and Total Quantity Sold of each Product as well')

# Calculate average unit price for each product
average_unit_price = data.groupby('Description')['UnitPrice'].mean()

# Calculate total quantity sold for each product
total_quantity_sold = data.groupby('Description')['Quantity'].sum()

# Display the calculated metrics
# print("Average Unit Price of each Product:")
# average_unit_price = pd.DataFrame(average_unit_price)
st.write("Average unit price:")
st.write(average_unit_price.sort_values(ascending=False).head(10))


# total_quantity_sold = pd.DataFrame(total_quantity_sold)
st.write("\nTotal Quantity Sold of each Product:")
st.write(total_quantity_sold.sort_values(ascending=False).head(10))

# Plot total quantity sold for each product
fig, ax = plt.subplots(figsize=(10, 6))
total_quantity_sold.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Products by Total Quantity Sold')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=90)
st.pyplot(fig)

# Plot total revenue generated for each product
fig, ax = plt.subplots(figsize=(10, 6))
average_unit_price.sort_values(ascending=False).head(10).plot(kind='bar')
# data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False)[:10].plot(kind='bar')
plt.title('Top 10 Products by Total Price Generated')
plt.xlabel('Product Description')
plt.ylabel('Total Revenue Generated')
plt.xticks(rotation=90)
st.pyplot(fig)

st.header('Quantifying the Impact of Discounts')
# Analyze the effect of discounts on sales
sales_with_discounts = data[data['DiscountedPrice'] < data['UnitPrice']]
sales_without_discounts = data[data['DiscountedPrice'] == data['UnitPrice']]

# Compare sales volume and revenue between discounted and non-discounted products
discounted_sales_volume = sales_with_discounts['Quantity'].sum()
non_discounted_sales_volume = sales_without_discounts['Quantity'].sum()

discounted_revenue = sales_with_discounts['TotalPrice'].sum()
non_discounted_revenue = sales_without_discounts['TotalPrice'].sum()

st.write("Sales Volume (Discounted):", discounted_sales_volume, 'units')
st.write("Sales Volume (Non-Discounted):", non_discounted_sales_volume, 'units')
st.write("Revenue (Discounted): $", round(discounted_revenue,2))
st.write("Revenue (Non-Discounted):$", round(non_discounted_revenue, 2))

st.write('Applying a 10% discount rate for quantities purchased of more than 10 units in the dataset implies that customers who purchase larger quantities are incentivized to buy even more due to the discount. This discount strategy encourages customers to increase their order size in order to take advantage of the discounted price. Consequently, it may lead to higher sales volumes and revenue, as customers are motivated to purchase more to benefit from the discount. Additionally, it can foster customer loyalty and satisfaction by offering them better value for their purchases.')

# Data
sales_volume = [discounted_sales_volume, non_discounted_sales_volume]
revenue = [discounted_revenue, non_discounted_revenue]
categories = ['Discounted', 'Non-Discounted']

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Sales Volume
axs[0].bar(categories, sales_volume, color=['blue', 'orange'])
axs[0].set_title('Sales Volume Comparison')
axs[0].set_ylabel('Sales Volume (units)')

# Revenue
axs[1].bar(categories, revenue, color=['blue', 'orange'])
axs[1].set_title('Revenue Comparison')
axs[1].set_ylabel('Revenue ($)')

plt.tight_layout()
st.pyplot(fig)

# Calculate the percentage change in sales volume
percentage_change_sales_volume = ((discounted_sales_volume - non_discounted_sales_volume) / non_discounted_sales_volume) 

# Calculate the percentage change in revenue
percentage_change_revenue = ((discounted_revenue - non_discounted_revenue) / non_discounted_revenue) 

st.write("Percentage Change in Sales Volume:", round(percentage_change_sales_volume,2), "%")
st.write("Percentage Change in Revenue:", round(percentage_change_revenue, 2), "%")

st.header("Discount Impact Analysis")

# Calculate revenue from discount and non-discount transactions
discount_transactions = data[data['DiscountedPrice'] < data['TotalPrice']]
non_discount_transactions = data[data['DiscountedPrice'] == data['TotalPrice']]

discount_revenue = discount_transactions['TotalPrice'].sum()
non_discount_revenue = non_discount_transactions['TotalPrice'].sum()

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
revenue_data = pd.DataFrame({'Category': ['Discounted', 'Non-Discounted'],
                             'Revenue': [discount_revenue, non_discount_revenue]})
sns.barplot(x='Category', y='Revenue', data=revenue_data, ax=ax)
ax.set_ylabel('Revenue')
ax.set_title('Revenue Comparison: Discounted vs. Non-Discounted Transactions')
st.pyplot(fig)

# Report findings
st.write(f"Revenue from Discount Transactions: ${discount_revenue:.2f}")
st.write(f"Revenue from Non-Discount Transactions: ${non_discount_revenue:.2f}")

# Section 3: Customer Segmentation Analysis
st.header("Customer Segmentation Analysis")
st.write('Use RFM (Recency, Frequency, Monetary) analysis for customer segmentation')

# Calculate RFM metrics for each customer
# Calculate Recency, Frequency, and Monetary values
today = pd.to_datetime('today')
rfm_data = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (today - x.max()).days,  # Recency
    'Quantity': 'sum',  # Frequency (assuming each row represents a purchase)
    'TotalPrice': 'sum'  # Monetary
}).rename(columns={
    'InvoiceDate': 'Recency',
    'Quantity': 'Frequency',
    'TotalPrice': 'Monetary'
})

# Visualize the RFM data distribution
fig, ax = plt.subplots(figsize=(12, 8))

plt.subplot(3, 1, 1)
sns.histplot(rfm_data['Recency'], kde=True)
plt.title('Recency Distribution')

plt.subplot(3, 1, 2)
sns.histplot(rfm_data['Frequency'], kde=True)
plt.title('Frequency Distribution')

plt.subplot(3, 1, 3)
sns.histplot(rfm_data['Monetary'], kde=True)
plt.title('Monetary Distribution')

plt.tight_layout()
st.pyplot(fig)

st.header('RFM metrics for each customer')

# Convert 'InvoiceDate' column to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Find the most recent purchase date
most_recent_date = data['InvoiceDate'].max()

# Calculate recency for each customer
data['Recency'] = most_recent_date - data.groupby('CustomerID')['InvoiceDate'].transform('max')

# Convert recency to number of days
data['Recency'] = data['Recency'].dt.days

# Displaying the first few rows of the dataframe to verify the results
st.write(data[['CustomerID', 'InvoiceDate', 'Recency']].head())

# Calculate the most recent purchase date for each customer
recency = data.groupby('CustomerID')['InvoiceDate'].max().reset_index()
# print(recency)
recency.columns = ['CustomerID', 'MostRecentPurchaseDate']

# Calculate the number of days since the most recent purchase
recency['Recency'] = (data['InvoiceDate'].max() - recency['MostRecentPurchaseDate']).dt.days

# Displaying the first few rows of the Recency dataframe
st.write("Recency:")
st.write(recency.head())

# Calculate Frequency (Number of purchases) for each customer
frequency = data.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequency.columns = ['CustomerID', 'Frequency']

# Calculate Monetary Value (Total amount spent) for each customer
monetary_value = data.groupby('CustomerID')['TotalPrice'].sum().reset_index()
monetary_value.columns = ['CustomerID', 'MonetaryValue']

# Displaying the first few rows of the Frequency and Monetary Value dataframes
st.write("Frequency:")
st.write(frequency.head())
st.write("\nMonetary Value:")
st.write(monetary_value.head())

# Merge Frequency, Monetary, and Recency dataframes
rfm_table = pd.merge(recency[['CustomerID', 'Recency']], frequency[['CustomerID', 'Frequency']], on='CustomerID')
rfm_table = pd.merge(rfm_table, monetary_value[['CustomerID', 'MonetaryValue']], on='CustomerID')

# Displaying the first few rows of the RFM table
st.write("RFM Table:")
st.write(rfm_table.head())

# Define function to segment customers based on RFM values
def segment_customers(row):
    if row['Recency'] <= recency_threshold and row['Frequency'] >= frequency_threshold and row['MonetaryValue'] >= monetary_threshold:
        return 'High Value'
    elif row['Recency'] <= recency_threshold and row['Frequency'] >= frequency_threshold:
        return 'Mid Value'
    else:
        return 'Low Value'

# Set thresholds for segmentation
recency_threshold = 90  # Customers who made a purchase within the last 90 days are considered "recent"
frequency_threshold = 3  # Customers who made at least 3 purchases are considered "frequent"
monetary_threshold = 1000  # Customers who spent at least $1000 are considered "high spenders"

# Apply segmentation function to each row in the RFM table
rfm_table['Segment'] = rfm_table.apply(segment_customers, axis=1)

# Displaying the segmented RFM table
st.write("Segmented RFM Table:")
st.write(rfm_table.head())


# Section 4: Outliers Analysis
st.header("Outliers Analysis")
# More analysis and visualizations here...

# Checking outliers in total price and quantity
total_price_zscore = (data['TotalPrice'] - data['TotalPrice'].mean()) / data['TotalPrice'].std()
quantity_zscore = (data['Quantity'] - data['Quantity'].mean()) / data['Quantity'].std()

# Defined a threshold for identifying outliers
# You can adjust this threshold based on your specific requirements
zscore_threshold = 3

# Find outliers based on the z-score
total_price_outliers = data[total_price_zscore.abs() > zscore_threshold]
quantity_outliers = data[quantity_zscore.abs() > zscore_threshold]

# Plotting box plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(data[total_price_zscore.abs() > zscore_threshold]['TotalPrice'])
plt.title('Box Plot for Total Price Outliers')
plt.ylabel('Total Price')
st.pyplot(fig)

# Plot box plot for Quantity outliers
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(data[quantity_zscore.abs() > zscore_threshold]['Quantity'])
plt.title('Box Plot for Quantity Outliers')
plt.ylabel('Quantity')
st.pyplot(fig)


# Display the outliers
st.write("Total Price Outliers:")
st.write(total_price_outliers)

st.write("\nQuantity Outliers:")
st.write(quantity_outliers)
st.write("""Outliers can be caused by various factors, including:

Data entry errors: Sometimes, outliers occur due to mistakes in data entry, such as typos or incorrect measurements.

Natural variation: In some cases, outliers are a result of natural variation in the data. For example, extreme weather conditions might lead to unusually high or low sales for certain products.

Measurement errors: Outliers can occur due to errors in measurement instruments or processes. For instance, a malfunctioning scale in a warehouse might lead to incorrect quantity measurements.

Rare events: Outliers may represent rare or unusual events that are not reflective of typical behavior. For example, a large one-time order from a new customer or a return of a defective product can result in outliers.

Skewed distribution: In datasets with skewed distributions, outliers may occur naturally at the tails of the distribution. These outliers may not necessarily indicate errors but rather extreme values within the dataset.""")

st.header('Month-Over-Month and Year-Over-Year growth rate in revenue for each product category and compare it to overall revenue growth')

# """
# This code will give us a DataFrame revenue_by_category_month containing the total revenue for each product category 
# for each month. We grouped the data by year, month, and product category, then calculated the sum of the 
# 'TotalPrice' column for each group. Finally, we reset the index to make the DataFrame more readable.
# # """
# Extract year and month from 'InvoiceDate' column
data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month

# Calculate total revenue for each product category for each month
revenue_by_category_month = data.groupby(['Year', 'Month', 'Description'])['TotalPrice'].sum().reset_index()

# Display the first few rows of the result
st.write(revenue_by_category_month.head())

st.header('Growth rate of revenue for each product category compared to the previous month or year')

# Extract year and month from 'InvoiceDate' column
data['Month'] = data['InvoiceDate'].dt.to_period('M')
data['Year'] = data['InvoiceDate'].dt.to_period('Y')

# Calculate total revenue for each month
# monthly_revenue = data.groupby('Month')['TotalPrice'].sum()
monthly_revenue = data.groupby([data['Month']])['TotalPrice'].sum()
# Calculate total revenue for each yearly
yearly_revenue = data.groupby([data['Year']])['TotalPrice'].sum()

# Calculate month-over-month growth rate in revenue
monthly_growth_rate = monthly_revenue.pct_change()
yearly_growth_rate = yearly_revenue.pct_change()

monthly_revenu_growth_rate = pd.merge(monthly_revenue, monthly_growth_rate, on='Month')
monthly_revenu_growth_rate.columns = ['Revenue', 'GrowthRate']
st.write(monthly_revenu_growth_rate)

st.header('Month-Over-Month growth rate')

# Plot month-over-month growth rate
fig, ax = plt.subplots(figsize=(10, 6))
monthly_growth_rate.plot(marker='o')
plt.title('Month-over-Month Growth Rate in Revenue', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Growth Rate', fontsize=14)
plt.grid(True)
st.pyplot(fig)

st.header('Year-Over-Year growth rate')

# Plot the overall growth rate using a bar chart
fig, ax = plt.subplots(figsize=(10, 6))
yearly_growth_rate.plot(kind='bar', color='skyblue')
plt.title('Year-Over-Year Revenue Growth Rate', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Growth Rate', fontsize=14)
plt.xticks(rotation=45, fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

st.write('Total revenue for each product category for each month and year')

# Group by product category and year
revenue_by_category_year = data.groupby(['Description', data['Year']])['TotalPrice'].sum()

# Group by product category and month
revenue_by_category_month = data.groupby(['Description', data['Month']])['TotalPrice'].sum()

st.write(revenue_by_category_year)

st.write(revenue_by_category_month)

# Conclusion
st.header("Conclusion")
st.write("Based on the analysis, it's evident that discounts have a significant impact on revenue. "
         "Further customer segmentation and outlier analysis provide valuable insights for improving "
         "marketing strategies and optimizing pricing strategies.")

# Footer
st.sidebar.title("About")
st.sidebar.info("This web app is created using Streamlit. It analyzes retail sales data "
                "to identify patterns and insights. " 
                "With Love from Yusuff Olatunji Sikiru")

