import openpyxl as px
import numpy as np
import csv
import re

workbook = px.load_workbook('training-Obama-Romney-tweets.xlsx')
sheet = workbook.get_sheet_by_name(name='Obama')

max_row = sheet.max_row

end = 'E'+ str(max_row)
csv_file = open('preprocessed.csv', 'w', newline='')
wr = csv.writer(csv_file, delimiter=',', quoting = csv.QUOTE_ALL)

stopwords = ['a', 'an', 'another', 'any', 'certain', 'each', 'every', 'her', 'his', 'i', 'its', 'its', 'my', '"no',
             'our', 'some', 'that', 'the', 'their', 'this', 'and', 'but', 'or', 'yet', 'for', 'nor', 'so', 'as',
             'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'around', 'at', 'before', 'behind',
             'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'down', 'during', 'except', 'following',
             'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'next', 'of', 'off', 'on',
             'onto', 'opposite', 'out', 'outside', 'over', 'past', 'plus', 'round', 'since', 'than', 'through', 'to',
             'toward', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'with', 'without']

def preprocess(cur_tweet):
    cur_tweet = cur_tweet.lower().strip()
    cur_tweet = re.sub(r'@[^\s]+', '', cur_tweet).strip()
    cur_tweet = re.sub(r'\#+[\w_]+[\w\'_\-]*[\w_]+', '', cur_tweet).strip()
    cur_tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', cur_tweet).strip()
    cur_tweet = re.sub(r'<[^>]+>', '', cur_tweet).strip()
    cur_tweet = re.sub(r'[\s]+', ' ', cur_tweet).strip()
    processed_tweet = []
    words = re.findall(r'[\w]+', cur_tweet)
    for word in words:
        if word not in stopwords:
            processed_tweet.append(word)

    final_tweet = " ".join(word for word in processed_tweet).strip()
    return final_tweet.strip()



for row in sheet['D3':end]:
    if row[1].value in [1,-1,0]:
        preproceesed_tweet = preprocess(str(row[1].value))
        if preproceesed_tweet.strip():
            tweets = [row[0].value, preproceesed_tweet]
            wr.writerow(tweets)


csv_file.close()



