from flask import Flask, render_template, request
#from wtforms import Form, TextAreaField, validators
import pickle
import json
from nltk.classify import *
import preprocess_data
import StringIO
import base64
import numpy as np
import matplotlib.pyplot as plt

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
'''
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
'''

#class HelloForm(Form):
#	sayhello = TextAreaField('',[validators.DataRequired()])
	
@app.route('/')
def index():
	#form = HelloForm(request.form)
	return render_template('first_app.html')
	
@app.route('/hello')
def hello():
	#form = HelloForm(request.form)
	#prediction = str(classify())
	count = 0
	countf = 0
	countm = 0
	countfe = 0
	countme = 0
	#count = 0
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
				if(label=="female"):
					countf = countf + 1
				else:
					countm = countm + 1
			else:
				if(label=="female"):
					countfe = countfe + 1
				else:
					countme = countme + 1
	#print prediction
	prediction = str(count/48.0)
	
	return render_template('hello.html',prediction=prediction,countf=countf,countm=countm,countfe=countfe,countme=countme)
@app.route('/plot_result')
def plot_result():
	img = StringIO.StringIO()
	y_test = np.array([0.48, 0.48, 0.72, 0.6, 0.48, 0.48, 0.68, 0.8, 0.84, 0.56])
	y_training = np.array([0.826666666667, 0.84, 0.826666666667, 0.835555555556, 0.84, 0.844444444444, 0.831111111111, 0.808888888889, 0.835555555556, 0.844444444444])
	x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	plt.plot(x, y_test)
	plt.plot(x, y_training)
	plt.xlabel('x axis label')
	plt.ylabel('y axis label')
	plt.title('Test and Training')
	plt.legend(['Test', 'Training'],loc='lower right')
	plt.savefig(img, format='png')
	img.seek(0)
	plot_url = base64.b64encode(img.getvalue())
	
	#form = HelloForm(request.form)
	#prediction = str(classify())
	#print prediction
	return render_template('plot_result.html',plot_url=plot_url)

	
if __name__ == '__main__':
	app.run(debug=True)
