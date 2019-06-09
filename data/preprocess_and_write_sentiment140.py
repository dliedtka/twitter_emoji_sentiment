import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../data/preprocessing/")
import preprocess_sentiment140
import pickle


# load sentiment140 data
sentiment140_data = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/sentiment140_tweets.pkl", "rb"))
#print (sentiment140_data["tweets"][:25])

# preprocess to tokenize usernames, URLs, remove repeat letters
new_tweets = preprocess_sentiment140.preprocess(sentiment140_data["tweets"]) # lowercase set true by default 
#print (new_tweets[:25])

# write
write_data = {}
write_data["tweets"] = new_tweets
write_data["labels"] = sentiment140_data["labels"]
#pickle.dump(write_data, open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/sentiment140_tweets_preprocessed_lowerTRUE.pkl", "wb"))
pickle.dump(write_data, open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/sentiment140_tweets_preprocessed.pkl", "wb"))

'''
# repeat with lowercase false
new_tweets = preprocess_sentiment140.preprocess(sentiment140_data["tweets"], lowercase=False)
#print (new_tweets[:25])

# write
write_data = {}
write_data["tweets"] = new_tweets
write_data["labels"] = sentiment140_data["labels"]
pickle.dump(write_data, open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/sentiment140_tweets_preprocessed_lowerFALSE.pkl", "wb"))
'''
