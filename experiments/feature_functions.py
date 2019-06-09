def unigrams_phi(tweet):
    '''
    Simply count unigrams in the input.
    '''
    feats = {}
    for word in tweet.split():
        if (word + "_UNIGRAM") not in feats.keys():
            feats[word + "_UNIGRAM"] = 1
        else:
            feats[word + "_UNIGRAM"] += 1

    return feats


def bigrams_phi(tweet):
    '''
    Simply count unigrams in the input.
    '''
    new_tweet = ["<START>"] + tweet.split() + ["<END>"]
    feats = {}
    for i in range(len(new_tweet) - 1):
        if (new_tweet[i] + "_" + new_tweet[i+1] + "_BIGRAM") not in feats.keys():
            feats[new_tweet[i] + "_" + new_tweet[i+1] + "_BIGRAM"] = 1
        else:
            feats[new_tweet[i] + "_" + new_tweet[i+1] + "_BIGRAM"] += 1

    return feats
