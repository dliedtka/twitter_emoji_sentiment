from sklearn.linear_model import SGDClassifier


class Support_Vector_Machine:
    '''
    Modified support vector machine class for initial fitting and then fine tuning of weights.
    '''

    def __init__(self, sent140_train_X, sent140_dev_X, sent140_train_Y, sent140_dev_Y, emoji_train_X, emoji_dev_X, emoji_test_X, emoji_train_Y, emoji_dev_Y, emoji_test_Y, emojiless_train_X, emojiless_dev_X, emojiless_test_X, emojiless_train_Y, emojiless_dev_Y, emojiless_test_Y, testing):
        '''
        Pass in initial data for fitting to constructor. Later adding passing SGD parameters into constructor.
        '''
        self.testing = testing

        self.sent140_train_X = sent140_train_X
        self.sent140_train_Y = sent140_train_Y
        self.sent140_dev_X = sent140_dev_X
        self.sent140_dev_Y = sent140_dev_Y

        self.emoji_train_X = emoji_train_X
        self.emoji_train_Y = emoji_train_Y
        self.emoji_dev_X = emoji_dev_X
        self.emoji_dev_Y = emoji_dev_Y
        if self.testing:
            self.emoji_test_X = emoji_test_X
            self.emoji_test_Y = emoji_test_Y
        
        self.emojiless_train_X = emojiless_train_X
        self.emojiless_train_Y = emojiless_train_Y
        self.emojiless_dev_X = emojiless_dev_X
        self.emojiless_dev_Y = emojiless_dev_Y
        if self.testing:
            self.emojiless_test_X = emojiless_test_X
            self.emojiless_test_Y = emojiless_test_Y

        # pass in model parameters for to constructor?
    

    def run_sent140(self):
        '''
        Trained on sent140, predict on emoji
        Report score on sent 140 too, just because it's intersting
        '''
        # model
        self.model_sent140 = SGDClassifier(random_state=69, warm_start=True)

        # train
        self.model_sent140.fit(self.sent140_train_X, self.sent140_train_Y)

        # test on sent140
        sent140_train_preds = self.model_sent140.predict(self.sent140_train_X)
        sent140_dev_preds = self.model_sent140.predict(self.sent140_dev_X)
        
        # test on emoji
        emoji_train_preds = self.model_sent140.predict(self.emoji_train_X)
        emoji_dev_preds = self.model_sent140.predict(self.emoji_dev_X)
        if self.testing:
            emoji_test_preds = self.model_sent140.predict(self.emoji_test_X)
        else:
            emoji_test_preds = None

        return (sent140_train_preds, sent140_dev_preds, emoji_train_preds, emoji_dev_preds, emoji_test_preds)
    

    def run_sent140_emojiless(self): 
        '''
        Trained on sent140, fine-tuned on emojiless, predict on emoji
        Report score on sent 140 too, just because it's intersting
        '''
        # model
        self.model_sent140_emojiless = SGDClassifier(random_state=69, warm_start=True)

        # train
        self.model_sent140_emojiless.fit(self.sent140_train_X, self.sent140_train_Y)
        self.model_sent140_emojiless.fit(self.emojiless_train_X, self.emojiless_train_Y)

        # test on sent140
        sent140_train_preds = self.model_sent140_emojiless.predict(self.sent140_train_X)
        sent140_dev_preds = self.model_sent140_emojiless.predict(self.sent140_dev_X)
        
        # test on emoji
        emoji_train_preds = self.model_sent140_emojiless.predict(self.emoji_train_X)
        emoji_dev_preds = self.model_sent140_emojiless.predict(self.emoji_dev_X)
        if self.testing:
            emoji_test_preds = self.model_sent140_emojiless.predict(self.emoji_test_X)
        else:
            emoji_test_preds = None

        return (sent140_train_preds, sent140_dev_preds, emoji_train_preds, emoji_dev_preds, emoji_test_preds)

    
    def run_sent140_emoji(self):
        '''
        Trained on sent140, fine-tuned on emoji, predict on emoji
        Report score on sent 140 too, just because it's intersting
        '''
        # model
        self.model_sent140_emoji = SGDClassifier(random_state=69, warm_start=True)

        # train
        self.model_sent140_emoji.fit(self.sent140_train_X, self.sent140_train_Y)
        self.model_sent140_emoji.fit(self.emoji_train_X, self.emoji_train_Y)

        # test on sent140
        sent140_train_preds = self.model_sent140_emoji.predict(self.sent140_train_X)
        sent140_dev_preds = self.model_sent140_emoji.predict(self.sent140_dev_X)
        
        # test on emoji
        emoji_train_preds = self.model_sent140_emoji.predict(self.emoji_train_X)
        emoji_dev_preds = self.model_sent140_emoji.predict(self.emoji_dev_X)
        if self.testing:
            emoji_test_preds = self.model_sent140_emoji.predict(self.emoji_test_X)
        else:
            emoji_test_preds = None
        
        return (sent140_train_preds, sent140_dev_preds, emoji_train_preds, emoji_dev_preds, emoji_test_preds)
        
