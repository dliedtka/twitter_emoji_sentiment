CS 224U Project: Twitter Emoji Sentiment Analysis
Dane Hankamer and David Liedtka


data/raw/sentiment140.csv is extracted from the download at https://www.kaggle.com/kazanova/sentiment140.

data/raw/emoji_sentiment_data.csv is extracted from the download at https://www.kaggle.com/thomasseleck/emoji-sentiment-data.

Run 'python data/read_sentiment140.py' to create data/usable/sentiment140_tweets.pkl. When loaded via pickle, this file is a dictionary that contains an array of Tweets and an array of labels from the Sentiment140 dataset. The index of each label corresponds to the index of each Tweet in the dataset. Write preprocessed versions (one version with all Tweets lower-cased, one with Tweets not lower-cased) with 'python data/preprocess_and_write_sentiment140.py' (creates data/usable/sentiment140_tweets_preprocess_lowerTRUE.pkl and data/usable/sentiment140_tweets_preprocess_lowerFALSE.pkl).

We ran 'python data/scrape_and_read_tweets.py' for about 10 hours on May 31, 2019 to create data/usable/emoji_tweets.pkl. We scraped live Tweets using Twitter's API that included any one of the top 25 emojis (ranked by number of appearances in Tweets in the Emoji Sentiment Data dataset). This dataset, our manually-generated emoji Tweet dataset, contains 541,030 Tweets.