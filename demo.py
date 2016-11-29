import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

def training_and_testing(candidate):
    input_tweets = csv.reader(open(candidate + '_training.csv', 'rt'), delimiter=',')
    unshuffled_tweets = list()

    for row in input_tweets:
        unshuffled_tweets.append(row)

    shuffled_tweets = shuffle(unshuffled_tweets, random_state=0)

    training_tweets_text = list()
    train_labels = list()

    for row in shuffled_tweets:
        sentiment = row[1]
        tweet = row[0]
        training_tweets_text.append(tweet)
        train_labels.append(sentiment)

    # reading the test tweets from testFile
    test_tweets = csv.reader(open(candidate + '_testing.csv', 'rt'), delimiter=',')
    test_tweets_text = list()
    test_labels = list()
    for row in test_tweets:
        tweet = row[0]
        test_tweets_text.append(tweet)
        test_labels.append(row[1])

    tf_idf_vectorizer = TfidfVectorizer()
    train_tf_idf_vector = tf_idf_vectorizer.fit_transform(training_tweets_text)
    test_tweets_vector = tf_idf_vectorizer.transform(test_tweets_text)

    #SVM
    svm_clf = svm.SVC(kernel='rbf', C=1.1, gamma=1.1)

    #SGDC classifier
    sgd_clf = linear_model.SGDClassifier()

    #knn classifier
    knn_clf = KNeighborsClassifier(n_neighbors=7)

    #logistic regression
    lr_clf = linear_model.LogisticRegression(C=1e5)

    #Naive Bayes
    nb_clf = BernoulliNB(alpha=.01)

    voting_clf = VotingClassifier(estimators=[('nb', nb_clf), ('svm', svm_clf), ('sgd', sgd_clf)], voting='hard')
    voting_clf = voting_clf.fit(train_tf_idf_vector, train_labels)
    voting_prediction = voting_clf.predict(test_tweets_vector)

    voting_accuracy = metrics.accuracy_score(test_labels, voting_prediction)
    confusion_metrics = metrics.confusion_matrix(test_labels, voting_prediction)

    print(confusion_metrics)
    print("Total accuracy for", candidate, "is", voting_accuracy * 100, "%")

    precisions = metrics.precision_score(test_labels, voting_prediction, average=None)
    recalls = metrics.recall_score(test_labels, voting_prediction, average=None)
    f1scores = metrics.f1_score(test_labels, voting_prediction, average=None)
    print("Precision for negative class ", precisions[0], " positive class ", precisions[2])
    print("Recall for negative class ", recalls[0], " positive class ", recalls[2]  )
    print("F1 or negative class ", f1scores[0], " positive class ", f1scores[2]  )
    print("finish\n-----------------------------\n\n")

training_and_testing('Obama')
training_and_testing('Romney')

