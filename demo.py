import csv
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from nltk.stem import PorterStemmer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import time

tweets = list()
input_tweets = csv.reader(open('preprocessed.csv', 'rt'), delimiter=',')
tweets_text = list()
unshuffled_tweets = list()

for row in input_tweets:
    unshuffled_tweets.append(row)

unshuffled_tweets = shuffle(unshuffled_tweets, random_state=0)

tweets_text = list()
tweet_labels = list()

for row in unshuffled_tweets:
    sentiment = row[1]
    tweet = row[0]
    tweets_text.append(tweet)
    tweet_labels.append(sentiment)
    feature_vector = [word.lower().strip() for word in tweet.split()]
    tweets.append((feature_vector, sentiment))

kf = KFold(n_splits=10)
fold_number = 1
total_accuracy = 0

#reading the test tweets from testFile
# test_tweets = csv.reader(open('preprocessed.csv', 'rt'), delimiter=',')
# testTweetsText = list()
# testTweetLabels = list()
# for row in test_tweets:
#     tweet = row[0]
#     testTweetsText.append(tweet)
#     testTweetLabels.append(row[1])
#     feature_vector = [word.lower().strip() for word in tweet.split()]

voting_fold_accuracies = list()
fold_number = 1
for train, test in kf.split(tweets):
    # Training
    start_time = time.time()
    training_tweets = [tweets[t] for t in train]

    training_tweets_text = [tweets_text[k] for k in train]
    test_tweets_text = [tweets_text[j] for j in test]
    train_labels = [tweets[t][1] for t in train]
    test_labels = [tweets[t][1] for t in test]

    tfidf_transformer = TfidfVectorizer()
    train_tfidf_vector = tfidf_transformer.fit_transform(training_tweets_text)
    X_new_tfidf = tfidf_transformer.transform(test_tweets_text)

    #SVM
    svm_clf = svm.SVC(kernel='linear', C=1.1, gamma=1.1)

    #SGDC classifier
    sgd_clf = linear_model.SGDClassifier()

    #knn classifier
    knn_clf = KNeighborsClassifier(n_neighbors=5)

    #logistic reression
    lr_clf = linear_model.LogisticRegression(C=1e5)

    #Naive Bayes
    nb_clf = BernoulliNB(alpha=.01)

    voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('nb', nb_clf), ('svm', svm_clf), ('sgd', sgd_clf), ('knn', knn_clf)], voting='soft')
    voting_clf = voting_clf.fit(train_tfidf_vector, train_labels)
    voting_prediction = voting_clf.predict(X_new_tfidf)

    voting_accuracy = metrics.accuracy_score(test_labels, voting_prediction)
    print("Fold accuracy for", fold_number, "is :=", voting_accuracy * 100, "%")
    voting_fold_accuracies.append(voting_accuracy)
    print("total time for fold", fold_number, "is", time.time() - start_time)
    fold_number += 1


aggregated_accuracy = 0;

for accuracy in voting_fold_accuracies:
    aggregated_accuracy += accuracy

print("Total accuracy is", aggregated_accuracy * 10, "%")
print("finish")

