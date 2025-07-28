import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

file_path = r"C:\Users\ayaan\OneDrive\Desktop\Unemployment in India.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

print("First 5 rows of data:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nNull values in each column:")
print(data.isnull().sum())

data.rename(columns={
    'Estimated Unemployment Rate (%)': 'Estimated Unemployment Rate',
    'Estimated Labour Participation Rate (%)': 'Estimated Labour Participation Rate',
    'Area': 'State'
}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

latest_date = data['Date'].max()
latest_data = data[data['Date'] == latest_date]

plt.figure(figsize=(12, 6))
sns.barplot(x='State', y='Estimated Unemployment Rate', data=latest_data)
plt.xticks(rotation=90)
plt.title(f'Unemployment Rate by State on {latest_date.date()}')
plt.tight_layout()
plt.show()

india_avg = data.groupby('Date')['Estimated Unemployment Rate'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(x='Date', y='Estimated Unemployment Rate', data=india_avg)
plt.title('Average Unemployment Rate in India Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

pivot_table = data.pivot_table(values='Estimated Unemployment Rate', index='State', columns='Date')

plt.figure(figsize=(15, 10))
sns.heatmap(pivot_table, cmap="YlOrRd", linewidths=0.5)
plt.title("Heatmap of Unemployment Rate by State Over Time")
plt.xlabel("Date")
plt.ylabel("State")
plt.tight_layout()
plt.show()
