import nltk
import csv
import preprocessor


#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

tweets = []
inpTweets = csv.reader(open('preprocessed.csv', 'rt'), delimiter=',')
featureList = []
i = 0
for row in inpTweets:
    i += 1
    sentiment = row[1]
    tweet = row[0]
    featureVector = [word.lower().strip() for word in tweet.split()]
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))
    if(i==500):
        break

print ("after for loop")
featureList = list(set(featureList))
#print (tweets)
#print (featureList)
print ("after for loop")

training_set = nltk.classify.util.apply_features(extract_features, tweets)
print ("traning")
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print ("classifier building")

# Test the classifier
testTweet = "obama debates ass cracker pataki"
processedTestTweet = preprocessor.preprocess(testTweet)
print (NBClassifier.classify(extract_features([word.lower().strip() for word in processedTestTweet.split()])))

testTweet = "love how polls are reported news when obama was lead they were saying how important they were now"
processedTestTweet =  preprocessor.preprocess(testTweet)
print (NBClassifier.classify(extract_features([word.lower().strip() for word in processedTestTweet.split()])))

testTweet = "smart women know when obama said policies would not cause energy prices will not skyrocket he meant it"
processedTestTweet =  preprocessor.preprocess(testTweet)
print (NBClassifier.classify(extract_features([word.lower().strip() for word in processedTestTweet.split()])))
