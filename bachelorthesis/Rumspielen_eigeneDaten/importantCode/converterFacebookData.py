import pandas as pd

data = pd.read_csv("/Users/tanjazast/Desktop/Bachelorarbeit/CSV/facebook_combined.txt"
                       , header=None)

data.columns = ['Source']

data.to_csv('/Users/tanjazast/Desktop/Bachelorarbeit/CSV/facebook_combined.csv',
                index=None)
