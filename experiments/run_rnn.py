import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../models/")
import rnn_classifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import emoji
from collections import Counter
import random


'''***********************************************************************************************************************
PARAMETERS
***********************************************************************************************************************'''

# dataset parameters *** (make these command line arguments?) ***

# data
#lowercase = True *** always true now ***

# features
# glove
glove25 = False
glove50 = False
glove100 = True
glove200 = False
if glove25:
    glove_path = "/../data/raw/glove.twitter.27B.25d.txt"
    glove_length = 25
if glove50:
    glove_path = "/../data/raw/glove.twitter.27B.50d.txt"
    glove_length = 50
elif glove100:
    glove_path = "/../data/raw/glove.twitter.27B.100d.txt"
    glove_length = 100
if glove200:
    glove_path = "/../data/raw/glove.twitter.27B.200d.txt"
    glove_length = 200
def glove2dict(src_filename):
    data = {}
    with open(src_filename) as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

# models
rnn = True
#lstm = False

counter = 0
if rnn:
    counter += 1
#if lstm: 
#    counter += 1
if counter != 1:
    print ("ERROR, exactly one model must be selected")
    sys.exit()

# testing (dev sets only evaluated if false)
testing = True

print ("parameters set")


'''***********************************************************************************************************************
LOAD DATA
***********************************************************************************************************************'''

emoji_scores = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/emoji_scores.pkl", "rb"))
emoji_list = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../data/usable/emoji_list.pkl", "rb"))

# load (preprocessed) sentiment140 data (preprocessed one time to save time with every run)
sent140_path = "/../data/usable/sentiment140_tweets_preprocessed.pkl"
sent140_data = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + sent140_path, "rb"))
sent140_length = len(sent140_data["labels"])

# load (preprocessed) emoji user-generated dataset
emoji_path = "/../data/usable/emoji_tweets_preprocessed_withEMOJI.pkl"
emoji_data = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + emoji_path, "rb"))
emoji_length = len(emoji_data["labels"])

# load emoji-less version of user-generated dataset (same labels, emojis have been stripped from tweets)
emojiless_path = "/../data/usable/emoji_tweets_preprocessed_withoutEMOJI.pkl"
emojiless_data = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + emojiless_path, "rb"))
emojiless_length = len(emojiless_data["labels"])

# sanity check
if len(emoji_data["tweets"]) != len(emojiless_data["tweets"]) or emoji_length != emojiless_length:
    print ("ERROR, emoji and emojiless mismatch")
    sys.exit()

print ("data loaded")


'''***********************************************************************************************************************
TRAIN / DEV / TEST
***********************************************************************************************************************'''

# split sentiment140 data into training (99%), and development (1%) sets
(sent140_train_X, sent140_dev_X, sent140_train_Y, sent140_dev_Y) = train_test_split(
    sent140_data["tweets"], sent140_data["labels"], test_size=0.01, random_state=69, shuffle=True)
# we obtain the following pos/neg splits (even enough for our purposes):
# train: 791911 pos, 792089 neg
# dev: 8089 pos, 7911 neg

# split emoji data into training (80%), development (10%), and test (10%) sets
(emoji_train_X, emoji_temp_X, emoji_train_Y, emoji_temp_Y) = train_test_split(
    emoji_data["tweets"], emoji_data["labels"], test_size=0.20, random_state=69, shuffle=True)
(emoji_dev_X, emoji_test_X, emoji_dev_Y, emoji_test_Y) = train_test_split(
    emoji_temp_X, emoji_temp_Y, test_size=0.50, random_state=69, shuffle=True)
# we obtain the following pos/neg splits (even enough for our purposes): *** check counts ***
# train: 5241 pos, 5319 neg
# dev: 676 pos, 644 neg
# test: 683 pos, 637 neg

# split emojiless data into training (80%), development (10%), and test (10%) sets, should match emoji splits
(emojiless_train_X, emojiless_temp_X, emojiless_train_Y, emojiless_temp_Y) = train_test_split(
    emojiless_data["tweets"], emojiless_data["labels"], test_size=0.20, random_state=69, shuffle=True)
(emojiless_dev_X, emojiless_test_X, emojiless_dev_Y, emojiless_test_Y) = train_test_split(
    emojiless_temp_X, emojiless_temp_Y, test_size=0.50, random_state=69, shuffle=True)
# we obtain the following pos/neg splits (even enough for our purposes):
# train: 5241 pos, 5319 neg
# dev: 676 pos, 644 neg
# test: 683 pos, 637 neg

# also split emoji data tweets into matching sets for later error analysis (just need dev and test)
(temp, emoji_temp_tweets, temp2, emoji_temp_Y) = train_test_split(
    emoji_data["tweets"], emoji_data["labels"], test_size=0.20, random_state=69, shuffle=True)
(emoji_dev_tweets, emoji_test_tweets, temp, temp2) = train_test_split(
    emoji_temp_tweets, emoji_temp_Y, test_size=0.50, random_state=69, shuffle=True)

print ("data split")


'''***********************************************************************************************************************
FEATURIZE
***********************************************************************************************************************'''

# glove embeddings loaded
print ("loading glove embeddings")
glove_lookup = glove2dict(os.path.dirname(os.path.abspath(__file__)) + glove_path)
print ("done")

# convert list of strings to list of lists
sent140_train_X_list = []
for tweet in sent140_train_X:
    sent140_train_X_list.append(tweet.split())
sent140_dev_X_list = []
for tweet in sent140_dev_X:
    sent140_dev_X_list.append(tweet.split())
emoji_train_X_list = []
for tweet in emoji_train_X:
    emoji_train_X_list.append(tweet.split())
emoji_dev_X_list = []
for tweet in emoji_dev_X:
    emoji_dev_X_list.append(tweet.split())
emoji_test_X_list = []
for tweet in emoji_test_X:
    emoji_test_X_list.append(tweet.split())
emojiless_train_X_list = []
for tweet in emojiless_train_X:
    emojiless_train_X_list.append(tweet.split())
emojiless_dev_X_list = []
for tweet in emojiless_dev_X:
    emojiless_dev_X_list.append(tweet.split())
emojiless_test_X_list = []
for tweet in emojiless_test_X:
    emojiless_test_X_list.append(tweet.split())


# borrowed from cs 224u utils.py
def randvec(n=50, lower=-0.5, upper=0.5):
    return np.array([random.uniform(lower, upper) for i in range(n)])

# borrowed from cs 224u utils.py
def get_vocab(X, n_words=None):
    wc = Counter([w for ex in X for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {w for w, c in wc}
    vocab.add("$UNK")
    return sorted(vocab)

# borrowed from cs 224u utils.py
def create_pretrained_embedding(lookup, vocab):
    vocab = sorted(set(lookup) & set(vocab))
    embedding = np.array([lookup[w] for w in vocab])
    if '$UNK' not in vocab:
        vocab.append("$UNK")
        embedding = np.vstack((embedding, randvec(embedding.shape[1])))
    return embedding, vocab

# vocabs
sent140_train_vocab = get_vocab(sent140_train_X_list, n_words=40000)
sent140_emojiless_train_vocab = get_vocab(sent140_train_X_list + emojiless_train_X_list, n_words=40000)
#sent140_train_vocab = get_vocab(sent140_train_X_list, n_words=10000)
#sent140_emojiless_train_vocab = get_vocab(sent140_train_X_list + emojiless_train_X_list, n_words=10000)

# embeddings
sent140_train_embedding, sent140_train_glove_vocab = create_pretrained_embedding(glove_lookup, sent140_train_vocab)
sent140_emojiless_train_embedding, sent140_emojiless_train_glove_vocab = create_pretrained_embedding(glove_lookup, sent140_emojiless_train_vocab)

# rewrite functions for emoji handling
# emoji substitution
def emoji_sub(emoji_word, lookup, dims):
    feats = np.zeros(dims)
    emoji_string = emoji.demojize(emoji_word)
    if emoji_string[0] != ":" or emoji_string[-1] != ":":
        return feats
    else:
        words = emoji_string[1:-1]
        counter = 0
        for word in words.split("_"):
            if word in lookup:
                feats += lookup[word]
                counter += 1
        if counter == 0:
            return feats
        else:
            return feats / counter

def get_vocab_emoji(X, used_emojis, n_words=None):
    wc = Counter([w for ex in X for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = []
    for pairing in wc:
        vocab.append(pairing[0])
    vocab.reverse()

    # insert emojis
    replace_idx = 0
    for emoji_ in used_emojis:
        if emoji_ not in vocab:
            # move replace index
            while vocab[replace_idx] in used_emojis:
                replace_idx += 1
            # substitute
            vocab[replace_idx] = emoji_

    vocab.append("$UNK")
    return sorted(vocab)

def create_pretrained_embedding_emoji(lookup, vocab, used_emojis, dims):
    new_lookup = lookup
    # add an embedding for each emoji
    for emoji_ in used_emojis:
        new_lookup[emoji_] = emoji_sub(emoji_, lookup, dims)

    vocab = sorted(set(new_lookup) & set(vocab))
    embedding = np.array([new_lookup[w] for w in vocab])
    if '$UNK' not in vocab:
        vocab.append("$UNK")
        embedding = np.vstack((embedding, randvec(embedding.shape[1])))
    return embedding, vocab


# list of used emojis
used_emojis = []
for entry in emoji_train_X_list: 
    for word in entry:
        if word in emoji_list:
            used_emojis.append(word)
used_emojis = set(used_emojis)

# ensure all emojis used in training set are in vocab
sent140_emoji_train_vocab = get_vocab_emoji(sent140_train_X_list + emoji_train_X_list, used_emojis, n_words=40000)
#sent140_emoji_train_vocab = get_vocab_emoji(sent140_train_X_list + emoji_train_X_list, used_emojis, n_words=10000)
# modified embedding
sent140_emoji_train_embedding, sent140_emoji_train_glove_vocab = create_pretrained_embedding_emoji(glove_lookup, sent140_emoji_train_vocab, used_emojis, glove_length)

print ("embeddings and vocabs generated")


'''***********************************************************************************************************************
MODELS
***********************************************************************************************************************'''

# use parametes to select a model type
model_name = ""
model = None

if rnn:
    model_name = "rnn"
    model = rnn_classifier.RNN_Classifier(sent140_train_X_list, sent140_dev_X_list, sent140_train_Y, sent140_dev_Y, sent140_train_embedding, sent140_train_glove_vocab, emoji_train_X_list, emoji_dev_X_list, emoji_test_X_list, emoji_train_Y, emoji_dev_Y, emoji_test_Y, sent140_emoji_train_embedding, sent140_emoji_train_glove_vocab, emojiless_train_X_list, emojiless_dev_X_list, emojiless_test_X_list, emojiless_train_Y, emojiless_dev_Y, emojiless_test_Y, sent140_emojiless_train_embedding, sent140_emojiless_train_glove_vocab, testing)
    
#if lstm:
#    model_name = "lstm"
#    model = None

else:
    pass

print ("model built")


'''***********************************************************************************************************************
RUN EXPERIMENTS
***********************************************************************************************************************'''

# store result information
results = {}
# store true labels
results["true"] = {}
results["true"]["sent140"] = {}
results["true"]["sent140"]["train"] = sent140_train_Y
results["true"]["sent140"]["dev"] = sent140_dev_Y
results["true"]["emoji"] = {}
results["true"]["emoji"]["train"] = emoji_train_Y
results["true"]["emoji"]["dev"] = emoji_dev_Y
results["true"]["emoji"]["test"] = emoji_test_Y

# first layer by model type
results["sent140"] = {}
results["sent140_emojiless"] = {}
results["sent140_emoji"] = {}

# sent140
print ("running sent140")
(sent140_sent140_train_preds, sent140_sent140_dev_preds, sent140_emoji_train_preds, sent140_emoji_dev_preds, sent140_emoji_test_preds) = model.run_sent140()
# store predictions
results["sent140"]["preds"] = {}
results["sent140"]["preds"]["sent140"] = {}
results["sent140"]["preds"]["sent140"]["train"] = sent140_sent140_train_preds
results["sent140"]["preds"]["sent140"]["dev"] = sent140_sent140_dev_preds
results["sent140"]["preds"]["emoji"] = {}
results["sent140"]["preds"]["emoji"]["train"] = sent140_emoji_train_preds
results["sent140"]["preds"]["emoji"]["dev"] = sent140_emoji_dev_preds
results["sent140"]["preds"]["emoji"]["test"] = sent140_emoji_test_preds
print ("done")
print (f1_score(emoji_test_Y, sent140_emoji_test_preds, pos_label=1))
pickle.dump(results, open("results_bkp.pkl", "wb"))

# sent140 + emojiless
print ("running sent140 + emojiless")
(sent140_emojiless_sent140_train_preds, sent140_emojiless_sent140_dev_preds, sent140_emojiless_emoji_train_preds, sent140_emojiless_emoji_dev_preds, sent140_emojiless_emoji_test_preds) = model.run_sent140_emojiless()
results["sent140_emojiless"]["preds"] = {}
results["sent140_emojiless"]["preds"]["sent140"] = {}
results["sent140_emojiless"]["preds"]["sent140"]["train"] = sent140_emojiless_sent140_train_preds
results["sent140_emojiless"]["preds"]["sent140"]["dev"] = sent140_emojiless_sent140_dev_preds
results["sent140_emojiless"]["preds"]["emoji"] = {}
results["sent140_emojiless"]["preds"]["emoji"]["train"] = sent140_emojiless_emoji_train_preds
results["sent140_emojiless"]["preds"]["emoji"]["dev"] = sent140_emojiless_emoji_dev_preds
results["sent140_emojiless"]["preds"]["emoji"]["test"] = sent140_emojiless_emoji_test_preds
print ("done")
print (f1_score(emoji_test_Y, sent140_emojiless_emoji_test_preds, pos_label=1))
pickle.dump(results, open("results_bkp.pkl", "wb"))

# sent140 + emoji
print ("running sent140 + emoji")
(sent140_emoji_sent140_train_preds, sent140_emoji_sent140_dev_preds, sent140_emoji_emoji_train_preds, sent140_emoji_emoji_dev_preds, sent140_emoji_emoji_test_preds) = model.run_sent140_emoji()
results["sent140_emoji"]["preds"] = {}
results["sent140_emoji"]["preds"]["sent140"] = {}
results["sent140_emoji"]["preds"]["sent140"]["train"] = sent140_emoji_sent140_train_preds
results["sent140_emoji"]["preds"]["sent140"]["dev"] = sent140_emoji_sent140_dev_preds
results["sent140_emoji"]["preds"]["emoji"] = {}
results["sent140_emoji"]["preds"]["emoji"]["train"] = sent140_emoji_emoji_train_preds
results["sent140_emoji"]["preds"]["emoji"]["dev"] = sent140_emoji_emoji_dev_preds
results["sent140_emoji"]["preds"]["emoji"]["test"] = sent140_emoji_emoji_test_preds
print ("done")
print (f1_score(emoji_test_Y, sent140_emoji_emoji_test_preds, pos_label=1))
pickle.dump(results, open("results_bkp.pkl", "wb"))

print ("experiments complete")


'''***********************************************************************************************************************
COMPUTE METRICS
***********************************************************************************************************************'''

# output string eventually written to file
results_string = ""

# iterate over models
for model in ["sent140", "sent140_emojiless", "sent140_emoji"]:
    
    # iterate over metrics
    for metric in ["f1", "precision", "recall"]:
        results[model][metric] = {}

        # iterate over pos/neg
        for label in ["pos", "neg"]:
            results[model][metric][label] = {}

            # iterate over dataset
            #for dataset in ["sent140", "emoji"]:
            for dataset in ["emoji"]:
                results[model][metric][label][dataset] = {}

                # iterate over dataset type
                for set_type in ["train", "dev", "test"]:
                    results[model][metric][label][dataset][set_type] = None

                    if not testing and set_type == "test":
                        continue

                    if set_type == "test" and dataset == "sent140":
                        continue

                    # true values to use
                    true = results["true"][dataset][set_type]
                    
                    # predicted values to use
                    preds = results[model]["preds"][dataset][set_type]

                    # determine pos label
                    if label == "pos":
                        pos_label = 1
                    elif label == "neg":
                        pos_label = 0
                    else:
                        pass

                    # which function to use
                    if metric == "f1":
                        score = f1_score(true, preds, pos_label=pos_label)
                    elif metric == "precision":
                        score = precision_score(true, preds, pos_label=pos_label)
                    elif metric == "recall":
                        score = recall_score(true, preds, pos_label=pos_label)
                    else:
                        pass

                    # write score
                    results[model][metric][label][dataset][set_type] = score

                    # add to string
                    results_string += model + " model, " + dataset + " data " + set_type + " set, " + metric + " score " + label + " labels: " + str(score) + "\n"

        results_string += "\n"
    results_string += "\n"

# compute accuracy separately
# iterate over models
for model in ["sent140", "sent140_emojiless", "sent140_emoji"]:
    
    # metric is accuracy
    results[model]["accuracy"] = {}
    # independent of pos/neg labels

    # iterate over dataset
    #for dataset in ["sent140", "emoji"]:
    for dataset in ["emoji"]:
        results[model]["accuracy"][dataset] = {}

        # iterate over dataset type
        for set_type in ["train", "dev", "test"]:
            results[model]["accuracy"][dataset][set_type] = None

            if not testing and set_type == "test":
                continue
            
            if set_type == "test" and dataset == "sent140":
                continue

            # true values to use
            true = results["true"][dataset][set_type]
                    
            # predicted values to use
            preds = results[model]["preds"][dataset][set_type]

            score = accuracy_score(true, preds)

            # write score
            results[model]["accuracy"][dataset][set_type] = score
            
            # add to string
            results_string += model + " model, " + dataset + " data " + set_type + " set, accuracy: " + str(score) + "\n"

        results_string += "\n"
    results_string += "\n"

print ("metrics computed")


'''***********************************************************************************************************************
WRITE RESULT FILES
***********************************************************************************************************************'''

result_filename = "/../results/"
if testing:
    result_filename += "test_results/"
result_filename += model_name + "_"
result_filename += str(glove_length) + "glove_"
if testing:
    result_filename += "testing_"
result_filename += "results.txt"

fout = open(os.path.dirname(os.path.abspath(__file__)) + result_filename, "w")

# write metrics
fout.write(results_string)

# write error analysis
fout.write("START ERROR ANALYSIS\n\n")
fout.write("format: model, correct/incorrect, true label, predicted label, tweet\n")

# just going to look at results on dev (or test) set
if testing:
    set_type = "test"
    tweets = emoji_test_tweets
else:
    set_type = "dev"
    tweets = emoji_dev_tweets

# iterate through models
for model in ["sent140", "sent140_emojiless", "sent140_emoji"]:

    true = results["true"][dataset][set_type]
    preds = results[model]["preds"][dataset][set_type]

    # assert same length
    if len(true) != len(preds) or len(true) != len(tweets):
        print ("ERROR writing file, mismatched true labels, predictions, and tweets")
        sys.exit()
    
    for i in range(len(tweets)):
        if true[i] == preds[i]:
            correct = "CORRECT"
        else:
            correct = "INCORRECT"

        fout.write(model + ", " + correct + ", true: " + str(true[i]) + ", pred: " + str(preds[i]) + ", tweet: " + tweets[i] + "\n")

fout.close()

print ("wrote file")
