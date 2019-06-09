import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/pytorch_models/")
from torch_rnn_classifier import TorchRNNClassifier
import numpy as np


class RNN_Classifier:
    '''
    Modified torch rnn classifier wrapper class for initial fitting and then fine tuning of weights.
    '''

    def __init__(self, sent140_train_X_list, sent140_dev_X_list, sent140_train_Y, sent140_dev_Y, sent140_train_embedding, sent140_train_glove_vocab, emoji_train_X_list, emoji_dev_X_list, emoji_test_X_list, emoji_train_Y, emoji_dev_Y, emoji_test_Y, sent140_emoji_train_embedding, sent140_emoji_train_glove_vocab, emojiless_train_X_list, emojiless_dev_X_list, emojiless_test_X_list, emojiless_train_Y, emojiless_dev_Y, emojiless_test_Y, sent140_emojiless_train_embedding, sent140_emojiless_train_glove_vocab, testing):
        '''
        Pass in initial data for fitting to constructor. Later adding passing logisitic regression
        parameters into constructor.
        '''
        self.testing = testing

        self.sent140_train_X = sent140_train_X_list
        self.sent140_train_Y = sent140_train_Y
        self.sent140_dev_X = sent140_dev_X_list
        self.sent140_dev_Y = sent140_dev_Y

        self.emoji_train_X = emoji_train_X_list
        self.emoji_train_Y = emoji_train_Y
        self.emoji_dev_X = emoji_dev_X_list
        self.emoji_dev_Y = emoji_dev_Y
        if self.testing:
            self.emoji_test_X = emoji_test_X_list
            self.emoji_test_Y = emoji_test_Y
        
        self.emojiless_train_X = emojiless_train_X_list
        self.emojiless_train_Y = emojiless_train_Y
        self.emojiless_dev_X = emojiless_dev_X_list
        self.emojiless_dev_Y = emojiless_dev_Y
        if self.testing:
            self.emojiless_test_X = emojiless_test_X_list
            self.emojiless_test_Y = emojiless_test_Y

        # embeddings and vocabs
        self.sent140_train_embedding = sent140_train_embedding
        self.sent140_train_glove_vocab = sent140_train_glove_vocab
        self.sent140_emoji_train_embedding = sent140_emoji_train_embedding
        self.sent140_emoji_train_glove_vocab = sent140_emoji_train_glove_vocab
        self.sent140_emojiless_train_embedding = sent140_emojiless_train_embedding
        self.sent140_emojiless_train_glove_vocab = sent140_emojiless_train_glove_vocab

        # pass in model parameters for to constructor?
    

    def run_sent140(self):
        '''
        Trained on sent140, predict on emoji
        Report score on sent 140 too, just because it's intersting
        '''
        # model
        #self.model_sent140 = TorchRNNClassifier(self.sent140_train_glove_vocab, embedding=self.sent140_train_embedding, bidirectional=True)
        self.model_sent140 = TorchRNNClassifier(self.sent140_train_glove_vocab, embedding=self.sent140_train_embedding)
        
        # train
        self.model_sent140.fit(self.sent140_train_X, self.sent140_train_Y)

        # test on sent140
        #sent140_train_preds = self.model_sent140.predict(self.sent140_train_X)
        #sent140_dev_preds = self.model_sent140.predict(self.sent140_dev_X)

        # test on emoji
        emoji_train_preds = self.model_sent140.predict(self.emoji_train_X)
        emoji_dev_preds = self.model_sent140.predict(self.emoji_dev_X)
        if self.testing:
            emoji_test_preds = self.model_sent140.predict(self.emoji_test_X)
        else:
            emoji_test_preds = None

        #return (sent140_train_preds, sent140_dev_preds, emoji_train_preds, emoji_dev_preds, emoji_test_preds)
        return (None, None, emoji_train_preds, emoji_dev_preds, emoji_test_preds)
    

    def run_sent140_emojiless(self):
        '''
        Trained on sent140, fine-tuned on emojiless, predict on emoji
        Report score on sent 140 too, just because it's intersting
        '''
        # model
        #self.model_sent140_emojiless = TorchRNNClassifier(self.sent140_emojiless_train_glove_vocab, embedding=self.sent140_emojiless_train_embedding, bidirectional=True)
        self.model_sent140_emojiless = TorchRNNClassifier(self.sent140_emojiless_train_glove_vocab, embedding=self.sent140_emojiless_train_embedding)
        
        # train
        # combine features
        combined_train_X = self.sent140_train_X + self.emojiless_train_X
        combined_train_Y = self.sent140_train_Y + self.emojiless_train_Y
        self.model_sent140_emojiless.fit(combined_train_X, combined_train_Y)
        
        # test on sent140
        #sent140_train_preds = self.model_sent140_emojiless.predict(self.sent140_train_X)
        #sent140_dev_preds = self.model_sent140_emojiless.predict(self.sent140_dev_X)
        
        # test on emoji
        emoji_train_preds = self.model_sent140_emojiless.predict(self.emoji_train_X)
        emoji_dev_preds = self.model_sent140_emojiless.predict(self.emoji_dev_X)        
        if self.testing:
            emoji_test_preds = self.model_sent140_emojiless.predict(self.emoji_test_X)
        else:
            emoji_test_preds = None
        
        #return (sent140_train_preds, sent140_dev_preds, emoji_train_preds, emoji_dev_preds, emoji_test_preds)
        return (None, None, emoji_train_preds, emoji_dev_preds, emoji_test_preds)

    
    def run_sent140_emoji(self):
        '''
        Trained on sent140, fine-tuned on emoji, predict on emoji
        Report score on sent 140 too, just because it's intersting
        '''
        # model
        #self.model_sent140_emoji = TorchRNNClassifier(self.sent140_emoji_train_glove_vocab, embedding=self.sent140_emoji_train_embedding, bidirectional=True)
        self.model_sent140_emoji = TorchRNNClassifier(self.sent140_emoji_train_glove_vocab, embedding=self.sent140_emoji_train_embedding)
        
        # train
        combined_train_X = self.sent140_train_X + self.emoji_train_X
        combined_train_Y = self.sent140_train_Y + self.emoji_train_Y
        self.model_sent140_emoji.fit(combined_train_X, combined_train_Y)

        # test on sent140
        #sent140_train_preds = self.model_sent140_emoji.predict(self.sent140_train_X)
        #sent140_dev_preds = self.model_sent140_emoji.predict(self.sent140_dev_X)
        
        # test on emoji
        emoji_train_preds = self.model_sent140_emoji.predict(self.emoji_train_X)
        emoji_dev_preds = self.model_sent140_emoji.predict(self.emoji_dev_X)
        if self.testing:
            emoji_test_preds = self.model_sent140_emoji.predict(self.emoji_test_X)
        else:
            emoji_test_preds = None
        
        #return (sent140_train_preds, sent140_dev_preds, emoji_train_preds, emoji_dev_preds, emoji_test_preds)
        return (None, None, emoji_train_preds, emoji_dev_preds, emoji_test_preds)
