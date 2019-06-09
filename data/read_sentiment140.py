import os
import csv
import pickle

data = {}
data["tweets"] = []
data["labels"] = []

with open(os.path.dirname(os.path.abspath(__file__)) + "/raw/sentiment140.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        data["tweets"].append(row[5])
        if int(row[0]) == 0:
            data["labels"].append(0)
        elif int(row[0]) == 4:
            data["labels"].append(1)
        else:
            raise ("invalid label")

pickle.dump(data, open(os.path.dirname(os.path.abspath(__file__)) + "/usable/sentiment140_tweets.pkl", "wb"))
