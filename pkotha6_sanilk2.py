import nltk
import csv
from sklearn.model_selection import KFold

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

tweets = []
inpTweets = csv.reader(open('preprocessed.csv', 'rt'), delimiter=',')
featureList = []
confusion_matrix = {}
confusion_matrix[0, 1] = 0
confusion_matrix[0, 0] = 0
confusion_matrix[0, -1] = 0
confusion_matrix[1, 0] = 0
confusion_matrix[1, 1] = 0
confusion_matrix[1, -1] = 0
confusion_matrix[-1, 0] = 0
confusion_matrix[-1, 1] = 0
confusion_matrix[-1, -1] = 0


for row in inpTweets:
    sentiment = row[1]
    tweet = row[0]
    featureVector = [word.lower().strip() for word in tweet.split()]
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))

featureList = list(set(featureList))
kf = KFold(n_splits=10)
fold_number = 1
total_accuracy = 0
for train, test in kf.split(tweets):
    # Training
    correct_count = 0
    training_tweets = [tweets[t] for t in train]
    training_set = nltk.classify.util.apply_features(extract_features, training_tweets)
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    # Testing
    for t in test:
        tweet_test = tweets[t][0]
        actual_class = tweets[t][1]
        classified_class = NBClassifier.classify(extract_features(tweet_test))
        print("Tweet : ", tweet_test)
        print("Actual class : ", actual_class)
        print("Classified class : ", classified_class)
        print("######################################################")
        if str(actual_class) == str(classified_class):
            correct_count += 1
        confusion_matrix[int(actual_class), int(classified_class)] += 1
    cur_fold_accuracy = correct_count / len(test)
    print("Accuracy for fold :", fold_number," -> ", cur_fold_accuracy)
    fold_number += 1
    print("######################################################")
    total_accuracy += cur_fold_accuracy

print("Total Classifier accuracy :", total_accuracy / 10 * 100)

print("Precision for positive class")
precision = confusion_matrix[1, 1] / confusion_matrix[1, 1] + confusion_matrix[0, 1] + confusion_matrix[-1, 1]

print("Recall for positive class")
recall = confusion_matrix[1, 1] / confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[1, -1]

print("F-score for positive class")
F_score = (2 * precision * recall) / (precision + recall)

print("Precision for negative class")
precision = confusion_matrix[-1, -1] / confusion_matrix[-1, -1] + confusion_matrix[0, -1] + confusion_matrix[1, -1]

print("Recall for negative class")
recall = confusion_matrix[-1, -1] / confusion_matrix[-1, -1] + confusion_matrix[-1, 0] + confusion_matrix[-1, 1]

print("F-score for negative class")
F_score = (2 * precision * recall) / (precision + recall)

print("Precision for mixed class")
precision = confusion_matrix[0, 0] / confusion_matrix[0, 0] + confusion_matrix[-1, 0] + confusion_matrix[1, 0]

print("Recall for mixed class")
recall = confusion_matrix[0, 0] / confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[0, -1]

print("F-score for mixed class")
F_score = (2 * precision * recall) / (precision + recall)

