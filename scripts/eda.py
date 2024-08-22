import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    #Load the dataset from the given file path
    return pd.read_csv(filepath)

def summary_statistics(data):
    #Generate summary statistics for the dataset
    return data.describe()

def plot_histogram(data, column_name):
    #Plot a histogram for the specified column
    sns.histplot(data[column_name])
    plt.show()

def main():
    filepath = '../data/benin-malanville.csv'
    data = load_data(filepath)
    
    print("Summary Statistics:")
    print(summary_statistics(data))
    
    print("Plotting GHI Histogram:")
    plot_histogram(data, 'GHI')

if __name__ == "__main__":
    main()
