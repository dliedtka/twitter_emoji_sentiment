#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import emoji
import json
import pickle
import os

#Variables that contains the user credentials to access Twitter API 
access_token = "1588599848-ZUkNTYw8iJ89od6ryN1lBm4dlPO37MIyl5ChvlB"
access_token_secret = "i5HJr50O5NZ6k63kjp5Zb9LWGytCEpclZY49B0eAFrRn2"
consumer_key = "P4sjwDQrSW0AwCgyaSvMjahBY"
consumer_secret = "dta4e5rFpd1aYOiuocZpACOisOvr9iqN4UXD7OVohxNktLFVIM"

tweets = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/usable/emoji_tweets.pkl", "rb"))
counter = 0
# all emojis featured in emoji sentiment data, filtered to 400 because of Twitter API query limits
emoji_list = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/usable/emoji_list.pkl", "rb"))[:400]

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        global tweets
        global counter
        global emoji_list

        def is_ascii_and_emoji(tweet):
            include = True
            for c in tweet:
                if not include:
                    break
                try:
                    c.encode("ascii")
                except:
                    # exception for emojis
                    if c not in emoji_list:
                        include = False
            return include

        #print (data)
        try:
            tweet = json.loads(data)

            # english, less than 140 characters, ascii only (besides emojis), contains an emoji
            if tweet["lang"] == "en" and len(tweet["text"]) <= 140 and is_ascii_and_emoji(tweet["text"]):
                counter += 1
                # print (counter)
                tweets.append(tweet["text"])
                # write to file every 1000 tweets
                if counter % 1000 == 0:
                    # eliminate duplicates
                    tweets = list(dict.fromkeys(tweets))
                    pickle.dump(tweets, open(os.path.dirname(os.path.abspath(__file__)) + "/usable/emoji_tweets.pkl", "wb"))
                    print ("wrote file " + str(int(counter / 1000)) + " times")
        except:
            pass

        return True

    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    #stream.filter(track=['python', 'javascript', 'ruby'])

    # trending topics
    #stream.filter(track=["ucla", "jay bruce", "bette midler", "#sdlive", "goldberg", "#wwessd", "#handmaidstale", "#thehandmaidstale", "#metromediajam", "angel hernandez", "straightprideparade", "gsellman", "mickey callaway", "archie bradley", "pedro severino", "datto", "paul walker", "scott kingery", "leigh ellis", "tanaka", "randy quaid"])

    # use as many emojis as possible (appears query limit is 400)
    stream.filter(track=emoji_list)
