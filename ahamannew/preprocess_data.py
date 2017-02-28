try:
    import json
except ImportError:
    import simplejson as json

import re
import nltk
from nltk.classify import *
import stopWord
#import pickle
#import os
#dest = os.path.join('genderprediction','pkl_objects')
#if not os.path.exists(dest):
#	os.makedirs(dest)
#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

def getfeatureVector(word, tweet):
    featureVector = []
    words = tweet.split()
    #print words[3]
    for singleword in words:
        singleword = stopWord.removeDuplicateWords(singleword)
        singleword = singleword.strip('\'"?,.!')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", singleword)
        if(singleword in word or val is None):
            continue
        else:
            featureVector.append(singleword.lower())
    return featureVector

#stopwords = stopWord.getStopWord('stopwords.txt')
#save_stopwords = open("E:\\univerHel\\my_github\\genderprediction\\pkl_objects\\stopwords.pkl","wb")
#pickle.dump(stopwords,save_stopwords)
#save_stopwords.close()
#pickle.dump(stopwords, open(os.path.join(dest,'stopwords.pkl'),'wb'),protocol=None)
#print stopwords[1:5]
'''
featureList = []
tweetFeature = []
with open('labelledtrainingData.txt', 'r') as f:
    for line in f:
        tweet = json.loads(line.strip())
        #print tweet['user']['gender']
        #print tweet['user']['name']
        #print tweet['user']['screen_name']
        #print tweet['user']['description']
        gender = tweet['user']['gender']
        #print gender
        
        tweet_text = processTweet(tweet['text'])
        #print tweet_text
        featureVector = getfeatureVector(stopwords, tweet_text)
        #print featureVector
        featureList.extend(featureVector)
        tweetFeature.append((featureVector, gender));
        #print tweetFeature
        
featureList = list(set(featureList))
#pickle.dump(featureList, open(os.path.join(dest,'featureList.pkl'),'wb'),protocol=None)

def extractFeatures(feature):
    features = set(feature)
    featureExtractList = {}
    for word in featureList:
        featureExtractList['contains(%s)'% word] = (word in features)
    return featureExtractList

training_set = nltk.classify.util.apply_features(extractFeatures, tweetFeature)
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
#pickle.dump(NBClassifier, open(os.path.join(dest,"NBClassifier.pkl"),"wb"),protocol=None)
#print nltk.classify.accuracy(NBClassifier, training_set)
save_classifiew = open("E:\\univerHel\\my_github\\genderprediction\\pkl_objects\\NBClassifier.pkl","wb")
pickle.dump(NBClassifier,save_classifiew)
save_classifiew.close()
'''
#NBClassifier.show_most_informative_features(20)


'''

##test part
count = 0
with open('labelledtestData.txt', 'r') as file:
    for line in file:
        tweet_test = json.loads(line.strip())
        #print tweet['user']['gender']
        #print tweet['user']['name']
        #print tweet['user']['screen_name']
        #print tweet['user']['description']
        gender = tweet_test['user']['gender']
        #print "right"
        #print gender
        
        tweet_text_test = processTweet(tweet_test['text'])
        #print tweet_text
        featureVector = getfeatureVector(stopwords, tweet_text_test)
        #print featureVector
        #featureList.extend(featureVector)
        #tweetFeature.append((featureVector, gender));
        label = NBClassifier.classify(extractFeatures(featureVector))
        #print label
        if(label==gender):
            count = count + 1
print count/49.0
'''


        #tweet_text = word_tokenize(tweet_text)
        #print tweet_text
        
        #print tweet['text']
        #hashtags = []
        #for hashtag in tweet['entities']['hashtags']:
        #    hashtags.append(hashtag['text'])
        #print hashtags
