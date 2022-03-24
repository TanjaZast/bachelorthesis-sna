import pandas as pd
import json

data = {
  "visited_Marketplace":[
        {
         "data": {
            "value": "19.04.2021"
          }
        },
        {
          "data": {
            "value": "14.04.2021"
          }
        },
        {
          "data": {
            "value": "25.03.2021"
          }
        },
        {
          "data": {
            "value": "23.03.2021"
          }
        },
        {
          "data": {
            "value": "19.04.2020"
          }
        },
        {
          "data": {
            "value": "10.09.2019"
          }
        },
        {
          "data": {
            "value": "06.10.2018"
          }
        },
        {
          "data": {
            "value": "17.07.2018"
          }
        },
        {
          "data": {
            "value": "29.06.2018"
          }
        },
        {
          "data": {
            "value": "08.06.2018"
          }
        },
        {
          "data": {
            "value": "07.06.2018"
          }
        },
        {
          "data": {
            "value": "13.05.2018"
          }
        }
      ]
    }


s1 = json.dumps(data)
d2 = json.loads(s1)

df = pd.json_normalize(d2["visited_Marketplace"])

df.to_csv("Visited_Marketplace.csv")
