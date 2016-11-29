import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

# Function which will take the name of the candidate and then opens their respective
# preprocessed csv files for training and testing

def training_and_testing(candidate):
    input_tweets = csv.reader(open(candidate + '_training.csv', 'rt'), delimiter=',')
    unshuffled_tweets = list()

    #Unshuffled Tweets
    for row in input_tweets:
        unshuffled_tweets.append(row)

    #shuffling training tweets
    shuffled_tweets = shuffle(unshuffled_tweets, random_state=0)

    training_tweets_text = list()
    train_labels = list()

    #Reading shuffled training tweets
    for row in shuffled_tweets:
        sentiment = row[1]
        tweet = row[0]
        training_tweets_text.append(tweet)
        train_labels.append(sentiment)

    # reading the test tweets from test file
    test_tweets = csv.reader(open(candidate + '_testing.csv', 'rt'), delimiter=',')
    test_tweets_text = list()
    test_labels = list()
    for row in test_tweets:
        tweet = row[0]
        test_tweets_text.append(tweet)
        test_labels.append(row[1])

    #Building TF-IDF Vectorizer for training and testing
    tf_idf_vectorizer = TfidfVectorizer()
    train_tf_idf_vector = tf_idf_vectorizer.fit_transform(training_tweets_text)
    test_tweets_vector = tf_idf_vectorizer.transform(test_tweets_text)

    #Building classifiers:
    #SVM
    svm_clf = svm.SVC(kernel='rbf', C=1.1, gamma=1.1)

    #SGDC classifier
    sgd_clf = linear_model.SGDClassifier(loss='log')

    #KNN classifier
    knn_clf = KNeighborsClassifier(n_neighbors=9)

    #logistic regression
    lr_clf = linear_model.LogisticRegression(C=1e5)

    #Naive Bayes
    nb_clf = BernoulliNB(alpha=.01)

    #Ensemble classifier, taking best 3 classifiers out of the above 5.
    voting_clf = VotingClassifier(estimators=[('nb', nb_clf), ('svm', svm_clf), ('sgd', sgd_clf)], voting='hard')
    voting_clf = voting_clf.fit(train_tf_idf_vector, train_labels)
    voting_prediction = voting_clf.predict(test_tweets_vector)

    #Ensemble accuracy
    voting_accuracy = metrics.accuracy_score(test_labels, voting_prediction)

    #Confusion matrix
    #confusion_metrics = metrics.confusion_matrix(test_labels, voting_prediction)

    #Printing metrics
    print("Analysis for", candidate)
    #print("\nConfusion Matrix\n", confusion_metrics)
    print("Overall accuracy for", candidate, "is %.2f" % (voting_accuracy * 100), "%")

    precisions = metrics.precision_score(test_labels, voting_prediction, average=None)
    recalls = metrics.recall_score(test_labels, voting_prediction, average=None)
    f1scores = metrics.f1_score(test_labels, voting_prediction, average=None)
    print("\nValues for negative class :\nPrecision:%.2f " % precisions[0], "Recall:%.2f " %recalls[0],
          "F1Score:%.2f " %f1scores[0])
    print("\nValues for positive class :\nPrecision:%.2f " % precisions[2], "Recall:%.2f " %recalls[2],
          "F1Score:%.2f " %f1scores[2])
    print("finish\n----------------------------------------------------------------------------------\n\n")

#Calling the functions for 2 candidates.
training_and_testing('Obama')
training_and_testing('Romney')

#End