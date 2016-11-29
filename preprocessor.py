import openpyxl as px
import csv
import re
from nltk.stem import PorterStemmer

candidate1 = 'Obama'
candidate2 = 'Romney'
training_workbook = 'training-Obama-Romney-tweets.xlsx'
testing_workbook = 'testing-Obama-Romney-tweets.xlsx'
stemmer = PorterStemmer()

stopWords = []
with open('StopWords.txt') as stop_file:
    stopWords = [word for word in stop_file]

def preprocess(cur_tweet):
    cur_tweet = cur_tweet.lower().strip()
    cur_tweet = cur_tweet.replace("\"", "")
    cur_tweet = re.sub(r'(@[^\s]+)', 'ATUSER', cur_tweet).strip()
    cur_tweet = re.sub(r'#([^\s]+)', r'\1', cur_tweet).strip()
    cur_tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', cur_tweet).strip()
    cur_tweet = re.sub(r'<[^>]+>', '', cur_tweet).strip()
    cur_tweet = re.sub(r'[\s]+', ' ', cur_tweet).strip()
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    cur_tweet = pattern.sub(r"\1\1", cur_tweet)
    processed_tweet = []
    words = re.findall(r'[\w]+', cur_tweet)
    for word in words:
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
        if word not in stopWords and val is not None and len(word) >= 2:
            if word != 'ATUSER':
                processed_tweet.append(stemmer.stem(word))

    final_tweet = " ".join(word for word in processed_tweet).strip()
    return final_tweet.strip()

def workbookDef(workbookName, sheetName, outputFile):
    workbook = px.load_workbook(workbookName)
    sheet = workbook.get_sheet_by_name(name=sheetName)

    max_row = sheet.max_row

    end = 'E' + str(max_row)
    csv_file = open(outputFile, 'w', newline='')
    wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)

    if(outputFile.endswith('training.csv')):
        starting_cell = 'D3'
    else :
        starting_cell = 'A3'
    for row in sheet[starting_cell:end]:
        if row[4].value in [1, -1, 0]:
            preprocessed_tweet = preprocess(str(row[0].value))
            if preprocessed_tweet.strip():
                tweets = [preprocessed_tweet, row[4].value]
                wr.writerow(tweets)

    csv_file.close()



#workbookDef(training_workbook, candidate1, candidate1 + '_training.csv')
workbookDef(testing_workbook, candidate1, candidate1 + '_testing.csv')
#workbookDef(training_workbook, candidate2, candidate2 + '_training.csv')
workbookDef(testing_workbook, candidate2, candidate2 + '_testing.csv')


