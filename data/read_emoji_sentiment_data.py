import os
import csv
import pickle

# list of emojis
data = []

with open(os.path.dirname(os.path.abspath(__file__)) + "/raw/emoji_sentiment_data.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        data.append(row[0])

    data = data[1:]
pickle.dump(data, open(os.path.dirname(os.path.abspath(__file__)) + "/usable/emoji_list.pkl", "wb"))


# dictionary of emojis to positivity score ((positive occurences - negative occurences) / total occurrences)
data = {}

with open(os.path.dirname(os.path.abspath(__file__)) + "/raw/emoji_sentiment_data.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    header = True
    for row in csv_reader:
        if header:
            header = False
            continue
        key = row[0]
        val = (int(row[6]) - int(row[4])) / int(row[2])
        data[key] = val

pickle.dump(data, open(os.path.dirname(os.path.abspath(__file__)) + "/usable/emoji_scores.pkl", "wb"))

