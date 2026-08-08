import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import missingno as msno

def plot_missing_values(df, title="Missing Values Matrix"):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Missing values matrix
    msno.matrix(df, ax=axes[0], color=(0.2, 0.4, 0.6))
    axes[0].set_title("Missing Values Matrix")
    
    # Missing values bar chart
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    if len(missing_percent) > 0:
        axes[1].bar(missing_percent.index, missing_percent.values, color='steelblue')
        axes[1].set_xlabel("Columns")
        axes[1].set_ylabel("Missing Percentage (%)")
        axes[1].set_title("Missing Values by Column")
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, "No Missing Values!", 
                    ha='center', va='center', fontsize=14)
        axes[1].set_title("Missing Values by Column")
    
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, columns=None, figsize=(15, 10)):
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            sns.boxplot(y=df[col], ax=axes[idx], color='steelblue')
            axes[idx].set_title(f"Outliers in {col}")
            axes[idx].set_ylabel(col)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_duplicates_summary(duplicate_report):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pie chart for duplicate distribution
    duplicate_count = duplicate_report.get('count', 0)
    total_rows = duplicate_report.get('total_rows', 0)
    duplicate_percent = duplicate_report.get('percentage', 0)
    
    values = [duplicate_count, total_rows - duplicate_count]
    labels = [f'Duplicates\n({duplicate_percent:.1f}%)', 
              f'Unique\n({100-duplicate_percent:.1f}%)']
    colors = ['#ff6b6b', '#4ecdc4']
    
    ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title("Duplicate Distribution")
    
    # Summary text
    ax2.text(0.1, 0.6, f"Total Rows: {total_rows}", fontsize=12)
    ax2.text(0.1, 0.4, f"Duplicates Found: {duplicate_count}", fontsize=12)
    ax2.text(0.1, 0.2, f"Duplicate Percentage: {duplicate_percent:.2f}%", fontsize=12)
    ax2.axis('off')
    ax2.set_title("Duplicate Summary")
    
    plt.tight_layout()
    plt.show()