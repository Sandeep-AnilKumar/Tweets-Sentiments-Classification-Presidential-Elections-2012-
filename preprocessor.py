import openpyxl as px
import csv
import re

workbook = px.load_workbook('training-Obama-Romney-tweets.xlsx')
sheet = workbook.get_sheet_by_name(name='Obama')

max_row = sheet.max_row

end = 'E' + str(max_row)
csv_file = open('preprocessed.csv', 'w', newline='')
wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)

stopWords = []
with open('StopWords.txt') as stop_file:
    stopWords = [word for word in stop_file]

def preprocess(cur_tweet):
    cur_tweet = cur_tweet.lower().strip()
    cur_tweet = cur_tweet.replace("\"", "")
    cur_tweet = re.sub(r'@[^\s]+', '', cur_tweet).strip()
    cur_tweet = re.sub(r'#([^\s]+)', r'\1', cur_tweet).strip()
    cur_tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', cur_tweet).strip()
    cur_tweet = re.sub(r'<[^>]+>', '', cur_tweet).strip()
    cur_tweet = re.sub(r'[\s]+', ' ', cur_tweet).strip()
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    cur_tweet = pattern.sub(r"\1\1", cur_tweet)
    processed_tweet = []
    words = re.findall(r'[\w]+', cur_tweet)
    for word in words:
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
        if word not in stopWords and val is not None and len(word) >= 2:
            processed_tweet.append(word)

    final_tweet = " ".join(word for word in processed_tweet).strip()
    return final_tweet.strip()



for row in sheet['D3':end]:
    if row[1].value in [1,-1,0]:
        preprocessed_tweet = preprocess(str(row[0].value))
        if preprocessed_tweet.strip():
            tweets = [preprocessed_tweet, row[1].value]
            wr.writerow(tweets)


csv_file.close()

