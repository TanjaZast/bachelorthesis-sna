import pandas as pd
a = pd.read_csv("/Users/tanjazast/Desktop/Bachelorarbeit/CSV/FacebookFriends.numbers")
print(a)
b = pd.read_csv("/Users/tanjazast/Desktop/Bachelorarbeit/CSV/Interaction.csv")
print(b)
#merged = a.merge(b, on='timestamp')
#merged
#merged.to_csv("output.csv", index=False)
