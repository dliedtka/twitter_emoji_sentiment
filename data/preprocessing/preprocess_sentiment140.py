import re
import string


def preprocess(tweets, lowercase=True):
    '''
    preprocess tweets to all lowercase (optional), tokenize usernames, URLs, and remove repeated letters; 
    return new array with preprocessed tweets in same order;
    convert to lowercase on by default, but their may be value in all caps for sentiment
    '''
  
    # new array to return
    new_tweets = []

    # iterate through all tweets
    counter = 0
    for tweet in tweets:
        counter += 1
        #print (counter)
        new_tweet = tweet

        # convert to all lowercase (optional)
        if lowercase:
            new_tweet = new_tweet.lower()

        # tokenize usernames (@____) to USERNAME
        new_tweet = re.sub(r'@\S+', "<user>", new_tweet)

        # tokenize URLs to URL
        # any URLs that start with http:// or https://
        new_tweet = re.sub(r'https?://\S+', "<url>", new_tweet)
        # any URLs missed above in the format www.____.___
        new_tweet = re.sub(r'www\.\S+\.\S+', "<url>", new_tweet)

        # remove repeated letters (3+ to 2), just iterate through 26 letters
        for c in string.ascii_lowercase:
            new_tweet = re.sub(r'' + c*3 + '+', "" + c*2, new_tweet)

        final_tweet = ""
        for c in new_tweet:
            if c == "#":
                final_tweet += " <hashtag> "
            else:
                final_tweet += c

        # preserve original tweet order
        new_tweets.append(final_tweet)

    return new_tweets
