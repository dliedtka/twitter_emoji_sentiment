import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import feature_functions
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../models/")
import maximum_entropy
import naive_bayes
import support_vector_machine
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


'''***********************************************************************************************************************
PARAMETERS
***********************************************************************************************************************'''

# dataset parameters *** (make these command line arguments?) ***

# data
#lowercase = True *** always true now ***

# features
unigrams = True
bigrams = True
normalize_feats = True 

# models
maxent = True
mnb = False
svm = False

counter = 0
if maxent:
    counter += 1
if mnb: 
    counter += 1
if svm:
    counter += 1
if counter != 1:
    print ("ERROR, exactly one model must be selected")
    sys.exit()

# testing (dev sets only evaluated if false)
testing = True

print ("parameters set")


'''***********************************************************************************************************************
LOAD DATA
***********************************************************************************************************************'''

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
FEATURIZE
***********************************************************************************************************************'''

# use parameters to turn feature functions on/off
# combine featurizing of sent140, emoji, and emojiless datasets so we vectorize to same features
# initialize as empty dicts for each tweet
combined_tweets = sent140_data["tweets"] + emoji_data["tweets"] + emojiless_data["tweets"]
combined_feats = []
for tweet in combined_tweets:
    combined_feats.append({})

# unigram features
if unigrams:
    counter = 0
    for tweet in combined_tweets:
        # merge old dict and new result dict
        print ("featurizing unigrams: " + str(counter))
        combined_feats[counter] = {**combined_feats[counter], **feature_functions.unigrams_phi(tweet)}
        counter += 1

# bigram features
if bigrams:
    counter = 0
    for tweet in combined_tweets:
        # merge old dict and new result dict
        print ("featurizing bigrams: " + str(counter))
        combined_feats[counter] = {**combined_feats[counter], **feature_functions.bigrams_phi(tweet)}
        counter += 1

# vectorize features
vectorizer = DictVectorizer()
combined_feats_vectorized = vectorizer.fit_transform(combined_feats)

# normalize features
if normalize_feats:
    combined_feats_vectorized = normalize(combined_feats_vectorized)    
    
# re-split into sent140, emoji, and emojiless
sent140_feats_vectorized = combined_feats_vectorized[:sent140_length]
emoji_feats_vectorized = combined_feats_vectorized[sent140_length:-emojiless_length]
emojiless_feats_vectorized = combined_feats_vectorized[-emojiless_length:]

print ("data featurized")


'''***********************************************************************************************************************
TRAIN / DEV / TEST
***********************************************************************************************************************'''

# split sentiment140 data into training (99%), and development (1%) sets
(sent140_train_X, sent140_dev_X, sent140_train_Y, sent140_dev_Y) = train_test_split(
    sent140_feats_vectorized, sent140_data["labels"], test_size=0.01, random_state=69, shuffle=True)
# we obtain the following pos/neg splits (even enough for our purposes):
# train: 791911 pos, 792089 neg
# dev: 8089 pos, 7911 neg

# split emoji data into training (80%), development (10%), and test (10%) sets
(emoji_train_X, emoji_temp_X, emoji_train_Y, emoji_temp_Y) = train_test_split(
    emoji_feats_vectorized, emoji_data["labels"], test_size=0.20, random_state=69, shuffle=True)
(emoji_dev_X, emoji_test_X, emoji_dev_Y, emoji_test_Y) = train_test_split(
    emoji_temp_X, emoji_temp_Y, test_size=0.50, random_state=69, shuffle=True)
# we obtain the following pos/neg splits (even enough for our purposes): *** check counts ***
# train: 5241 pos, 5319 neg
# dev: 676 pos, 644 neg
# test: 683 pos, 637 neg

# split emojiless data into training (80%), development (10%), and test (10%) sets, should match emoji splits
(emojiless_train_X, emojiless_temp_X, emojiless_train_Y, emojiless_temp_Y) = train_test_split(
    emojiless_feats_vectorized, emojiless_data["labels"], test_size=0.20, random_state=69, shuffle=True)
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
MODELS
***********************************************************************************************************************'''

# use parametes to select a model type
model_name = ""
model = None

if maxent:
    model_name = "maxent"
    model = maximum_entropy.Maximum_Entropy(sent140_train_X, sent140_dev_X, sent140_train_Y, sent140_dev_Y, emoji_train_X, emoji_dev_X, emoji_test_X, emoji_train_Y, emoji_dev_Y, emoji_test_Y, emojiless_train_X, emojiless_dev_X, emojiless_test_X, emojiless_train_Y, emojiless_dev_Y, emojiless_test_Y, testing)

if mnb:
    model_name = "mnb"
    model = naive_bayes.Naive_Bayes(sent140_train_X, sent140_dev_X, sent140_train_Y, sent140_dev_Y, emoji_train_X, emoji_dev_X, emoji_test_X, emoji_train_Y, emoji_dev_Y, emoji_test_Y, emojiless_train_X, emojiless_dev_X, emojiless_test_X, emojiless_train_Y, emojiless_dev_Y, emojiless_test_Y, testing)

if svm:
    model_name = "svm"
    model = support_vector_machine.Support_Vector_Machine(sent140_train_X, sent140_dev_X, sent140_train_Y, sent140_dev_Y, emoji_train_X, emoji_dev_X, emoji_test_X, emoji_train_Y, emoji_dev_Y, emoji_test_Y, emojiless_train_X, emojiless_dev_X, emojiless_test_X, emojiless_train_Y, emojiless_dev_Y, emojiless_test_Y, testing)

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
            for dataset in ["sent140", "emoji"]:
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
    for dataset in ["sent140", "emoji"]:
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
if unigrams:
    result_filename += "unigrams_"
if bigrams:
    result_filename += "bigrams_"
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
