import pandas as pd


semeval_test_qa = pd.read_csv("data/test/test_qa.csv")
for i, row in semeval_test_qa.iterrows():
    print(row)
    break