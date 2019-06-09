# for documentation on vader sentiment analysis:
# https://github.com/cjhutto/vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random


def generate_labels(tweets):
    
    positive_tweets = []
    negative_tweets = []
    using_tweets = []
    labels = []
    #scores = []

    analyzer = SentimentIntensityAnalyzer()

    counter = 0
    for tweet in tweets:
        score = analyzer.polarity_scores(tweet)["compound"]
        print ("generating labels: " + str(counter))
        counter += 1
        
        #scores.append(score)

        # roughly 6,600 tweets score below -0.7, 35,000 above 0.7, so randomly sample 6,600 above 0.7 to maintain balance
        # if we use 0.7 and -0.7, we get roughly 40,000 and 5,000
        if score > 0.7:
            positive_tweets.append(tweet)
        elif score < -0.7:
            negative_tweets.append(tweet)
        else:
            pass

    # pick random 6,600 positive, negative tweets
    # shuffle positive and negative tweets
    random.shuffle(positive_tweets)
    random.shuffle(negative_tweets)

    for i in range(6600):
        using_tweets.append(negative_tweets[i])
        labels.append(0)

    for i in range(6600):
        using_tweets.append(positive_tweets[i])
        labels.append(1)

    return (using_tweets, labels)

    '''
    counter = 0
    for score in scores:
        counter += 1
    print ("total: " + str(counter))

    counter = 0
    for score in scores:
        if score > 0.9:
            counter += 1
    print ("above 0.9: " + str(counter))

    counter = 0
    for score in scores:
        if score < -0.9:
            counter += 1
    print ("below -0.9: " + str(counter))

    counter = 0
    for score in scores:
        if score > 0.8:
            counter += 1
    print ("above 0.8: " + str(counter))

    counter = 0
    for score in scores:
        if score < -0.8:
            counter += 1
    print ("below -0.8: " + str(counter))

    counter = 0
    for score in scores:
        if score > 0.7:
            counter += 1
    print ("above 0.7: " + str(counter))

    counter = 0
    for score in scores:
        if score < -0.7:
            counter += 1
    print ("below -0.7: " + str(counter))
    '''
