import pandas as pd

data = pd.read_csv("/Users/tanjazast/Downloads/twitter_combined (1).txt"
                       , header=None)

data.columns = ['Source']

data.to_csv('/Users/tanjazast/Desktop/Bachelorthesis/bachelorthesis-sna/bachelorthesis/CSVtwitter_combined.csv',
                index=None)
