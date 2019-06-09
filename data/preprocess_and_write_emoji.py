import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../data/preprocessing/")
import preprocess_emoji
import generate_labels_emoji
import pickle


def remove_emojis(tweets):
    '''
    Strip emojis from a list of tweets
    '''
    emoji_list = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/usable/emoji_list.pkl", "rb"))

    new_tweets = []
    counter = 0
    for tweet in tweets:
        print ("removing emojis: " + str(counter))
        counter += 1
        new_tweet = ""
        for c in tweet:
            if c in emoji_list:
                pass
            else:
                new_tweet += c
        new_tweets.append(new_tweet)

    return new_tweets


# load emoji tweets
emoji_tweets = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/usable/emoji_tweets.pkl", "rb"))
print (len(emoji_tweets))
#print (emoji_tweets[:25])

# preprocess, first maintaining emojis
preprocessed_tweets = preprocess_emoji.preprocess(emoji_tweets) # lowercase set true by default 
#print (new_tweets[:25])

# write
write_data = {}
(write_data["tweets"], write_data["labels"]) = generate_labels_emoji.generate_labels(preprocessed_tweets) # vader labels
pickle.dump(write_data, open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/emoji_tweets_preprocessed_withEMOJI.pkl", "wb"))

# now preprocess and strip out emojis, use same labels
emojiless_tweets = remove_emojis(write_data["tweets"])

# write
write_data["tweets"] = emojiless_tweets
pickle.dump(write_data, open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/emoji_tweets_preprocessed_withoutEMOJI.pkl", "wb"))

# not worrying about lowercase false anymore
