import csv

tweets = []
inpTweets = csv.reader(open('preprocessed.csv', 'rt'), delimiter=',')
i = 0
for row in inpTweets:
    i += 1
    sentiment = row[1]
    tweet = row[0]
    featureVector = [word.lower().strip() for word in tweet.split()]
    tweets.append((featureVector, sentiment))
    if i == 10:
        break

print(tweets)

