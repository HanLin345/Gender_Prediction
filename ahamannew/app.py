from flask import Flask, render_template, request
#from wtforms import Form, TextAreaField, validators
import pickle
import json
from nltk.classify import *
import preprocess_data

stopwords_f = open("stopwords.pkl","rb")
stopwords = pickle.load(stopwords_f)
stopwords_f.close()
NBClassifier_f = open("NBClassifier.pkl","rb")
NBClassifier = pickle.load(NBClassifier_f)
NBClassifier_f.close()
featureList_f = open("featureList.pkl","rb")
featureList = pickle.load(featureList_f)
featureList_f.close()

app = Flask(__name__)

def extractFeatures(feature):
    features = set(feature)
    featureExtractList = {}
    for word in featureList:
        featureExtractList['contains(%s)'% word] = (word in features)
    return featureExtractList

def classify():
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
        
			tweet_text_test = preprocess_data.processTweet(tweet_test['text'])
        #print tweet_text
			featureVector = preprocess_data.getfeatureVector(stopwords, tweet_text_test)
        #print featureVector
        #featureList.extend(featureVector)
        #tweetFeature.append((featureVector, gender));
			label = NBClassifier.classify(extractFeatures(featureVector))
        #print label
			if(label==gender):
				count = count + 1
	return count/49.0
	

#class HelloForm(Form):
#	sayhello = TextAreaField('',[validators.DataRequired()])
	
@app.route('/')
def index():
	#form = HelloForm(request.form)
	return render_template('first_app.html')
	
@app.route('/hello')
def hello():
	#form = HelloForm(request.form)
	prediction = str(classify())
	#print prediction
	return render_template('hello.html',prediction=prediction)

	
if __name__ == '__main__':
	app.run(debug=True)
