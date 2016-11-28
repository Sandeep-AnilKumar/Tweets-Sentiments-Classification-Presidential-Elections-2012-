import nltk
import csv
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
import time

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
tweetsText = []
unshuffledTweets = []

for row in inpTweets:
    unshuffledTweets.append(row)

#print("Before shuffling", unshuffledTweets[:10])

unshuffledTweets = shuffle(unshuffledTweets, random_state=0)

#print("After shuffling", unshuffledTweets[:10])

i = 0

tweetsText = []
tweetLabels = []

classifier_accuracies = [0, 0, 0, 0, 0]
i=0

for row in unshuffledTweets:
    i += 1
    sentiment = row[1]
    tweet = row[0]
    tweetsText.append(tweet)
    tweetLabels.append(sentiment)
    featureVector = [word.lower().strip() for word in tweet.split()]
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))
    # if(i==1000):
    #     break

featureList = list(set(featureList))
kf = KFold(n_splits=10)
fold_number = 1
total_accuracy = 0

#reading the test tweets from testFile
testTweets = csv.reader(open('preprocessed.csv', 'rt'), delimiter=',')
i=0
testTweetsText =[]
testTweetLabels =[]
for row in testTweets:
    i += 1
    tweet = row[0]
    testTweetsText.append(tweet)
    testTweetLabels.append(row[1])
    featureVector = [word.lower().strip() for word in tweet.split()]

for train, test in kf.split(tweets):
    # Training
    start_time = time.time()
    nb_correct_count = 0
    svm_correct_count = 0
    knn_correct_count = 0
    sgdc_correct_count = 0
    logreg_correct_count = 0
    mlp_correct_count = 0
    voted_correct_count = 0
    training_tweets = [tweets[t] for t in train]

    count_vect = CountVectorizer()
    training_tweets_text = [tweetsText[k] for k in train]
    test_tweets_text = [tweetsText[j] for j in test]
    train_labels = [tweets[t][1] for t in train]
    test_labels = [tweets[t][1] for t in test]

    tfidf_transformer = TfidfVectorizer()
    train_tfidf_vector = tfidf_transformer.fit_transform(training_tweets_text)

    predictions = []
    clf = svm.SVC(kernel='linear', C=1.1, gamma=1.1)
    clf.fit(train_tfidf_vector, train_labels)
    X_new_tfidf = tfidf_transformer.transform(test_tweets_text)
    svmPredict = clf.predict(X_new_tfidf)
    predictions.append(svmPredict)
    # print(svm.score(test_tweets_tfidf_vector, test_labels))
    # print(confusion_matrix(pred, test_labels))

    #SGDC classifier
    sgdclf = linear_model.SGDClassifier()
    sgdclf.fit(train_tfidf_vector, train_labels)
    sgdcPredict = sgdclf.predict(X_new_tfidf)

    #knn classifier
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_tfidf_vector, train_labels)
    knnPredict = neigh.predict(X_new_tfidf)
    predictions.append(knnPredict)

    #logistic reression
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train_tfidf_vector, train_labels)
    logregPredict = logreg.predict(X_new_tfidf)
    predictions.append(logregPredict)

    #multilayer perceptron
    # mlp = MLPClassifier(activation='tanh',learning_rate_init=0.01,  max_iter=10000, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 50, 10), random_state=1)
    # mlp.fit(train_tfidf_vector, train_labels)
    # mlpregPredict = mlp.predict(X_new_tfidf)

    #Naive Bayes
    training_set = nltk.classify.util.apply_features(extract_features, training_tweets)
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

    print("in some fold, training")
    # Testing
    index = -1
    debug = False
    for t in test:
        index += 1
        tweet_test = tweets[t][0]
        actual_class = tweets[t][1]
        nbPredict = NBClassifier.classify(extract_features(tweet_test))
        #taking voting#
        predictedLabels = [nbPredict, svmPredict[index], sgdcPredict[index], knnPredict[index], logregPredict[index]]
        votedLabel =  max(set(predictedLabels), key=predictedLabels.count)

        if (debug):
            # print("Tweet : ", tweet_test)
            print("Actual class : ", actual_class)
            print("Classified class : ", nbPredict)
            print("svm prediction: ", svmPredict[index])
            print("sgdc prediction: ", sgdcPredict[index])
            print("knn prediction: ", knnPredict[index])
            print("log regression prediction: ", logregPredict[index])
            print("voted prediction: ", votedLabel)
            print("######################################################")

        if str(actual_class) == str(votedLabel):
            voted_correct_count += 1

        if str(actual_class) == str(svmPredict[index]):
            svm_correct_count += 1

        if str(actual_class) == str(sgdcPredict[index]):
            sgdc_correct_count += 1

        if str(actual_class) == str(knnPredict[index]):
            knn_correct_count += 1

        if str(actual_class) == str(logregPredict[index]):
            logreg_correct_count += 1

        if str(actual_class) == str(nbPredict):
            nb_correct_count += 1

        # if str(actual_class) == str(mlpregPredict[index]):
        #     mlp_correct_count += 1
        confusion_matrix[int(actual_class), int(votedLabel)] += 1

    print ("elapsed time :", (time.time() - start_time)/60)
    nb_cur_fold_accuracy = nb_correct_count / len(test)
    classifier_accuracies[0] += nb_correct_count / len(test)

    svm_cur_fold_accuracy = svm_correct_count / len(test)
    classifier_accuracies[1] += svm_correct_count / len(test)

    sgdc_cur_fold_accuracy = sgdc_correct_count / len(test)
    classifier_accuracies[2] += sgdc_correct_count / len(test)

    knn_cur_fold_accuracy = knn_correct_count / len(test)
    classifier_accuracies[3] += knn_correct_count / len(test)

    logreg_cur_fold_accuracy = logreg_correct_count / len(test)
    classifier_accuracies[4] += logreg_correct_count / len(test)


    voted_cur_fold_accuracy = voted_correct_count / len(test)
    #mlp_cur_fold_accuracy = mlp_correct_count / len(test)

    print("naive bayes Accuracy for fold :", fold_number," -> ", nb_cur_fold_accuracy, )
    print("svm Accuracy for fold :", fold_number, " -> ", svm_cur_fold_accuracy)
    print("sgdc Accuracy for fold :", fold_number, " -> ", sgdc_cur_fold_accuracy)
    print("knn Accuracy for fold :", fold_number, " -> ", knn_cur_fold_accuracy)
    print("logreg Accuracy for fold :", fold_number, " -> ", logreg_cur_fold_accuracy)
    print("voted Accuracy for fold :", fold_number, " -> ", voted_cur_fold_accuracy)
    #print("neural Accuracy for fold :", fold_number, " -> ", mlp_cur_fold_accuracy)
    fold_number += 1
    print("######################################################")
    total_accuracy += voted_cur_fold_accuracy

print("Total Classifier accuracy :", total_accuracy / 10 * 100)

classifiers = ["nb", "svm", "sgd", "knn", "log"]
for classifier_accuracy in classifier_accuracies:
    print(classifier_accuracy)

for classifier, classifier_accuracy in zip(classifiers, classifier_accuracies):
    print("accuracy for ", classifier, " is -> ", classifier_accuracy/10)


def printMetrics():
    print("Confusion Matrix")
    print(confusion_matrix)

    print("Precision for positive class:", end='')
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1] + confusion_matrix[-1, 1])
    print(precision)

    print("Recall for positive class:", end='')
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[1, -1])
    print(recall)

    print("F-score for positive class:", end='')
    F_score = (2 * precision * recall) / (precision + recall)
    print(F_score)

    print("Precision for negative class:", end='')
    precision = confusion_matrix[-1, -1] / (confusion_matrix[-1, -1] + confusion_matrix[0, -1] + confusion_matrix[1, -1])
    print(precision)

    print("Recall for negative class:", end='')
    recall = confusion_matrix[-1, -1] / (confusion_matrix[-1, -1] + confusion_matrix[-1, 0] + confusion_matrix[-1, 1])
    print(recall)

    print("F-score for negative class:", end='')
    F_score = (2 * precision * recall) / (precision + recall)
    print(F_score)

    print("Precision for mixed class:", end='')
    precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[-1, 0] + confusion_matrix[1, 0])
    print(precision)

    print("Recall for mixed class:", end='')
    recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[0, -1])
    print(recall)

    print("F-score for mixed class:", end='')
    F_score = (2 * precision * recall) / (precision + recall)
    print(F_score)

printMetrics()
