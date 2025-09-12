import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  
df = pd.read_csv("data.csv")  
for col in ["healthRiskScore","windgust","dew"]:
    print(df[col].describe())
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
# Compute correlation matrix
corr = df[['healthRiskScore','windgust','dew']].corr()

# Plot heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
