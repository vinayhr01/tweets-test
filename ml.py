import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import snscrape.modules.twitter as sntwitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from collections import Counter

dataset = pd.read_csv('final.csv', encoding_errors='ignore')#, low_memory=False)

def sentiment(t):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(t)

    if sentiment_dict['compound'] >= 0.05 :
        return sentiment_dict['compound']
    elif sentiment_dict['compound'] <= - 0.05 :
        return sentiment_dict['compound']
 
    else :
        return sentiment_dict['compound']

def addCapTag(word):
    """ Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_ """
    if(len(re.findall("[A-Z]{3,}", word))):
        word = word.replace('\\', '' )
        transformed = re.sub("[A-Z]{3,}", "ALL_CAPS_"+word, word)
        return transformed
    else:
        return word

def cleaning(text):
    text = text.lower()
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', ' ', text)       
    text = re.sub(r'[^\x00-\x7f]',r' ',text)
    text = re.sub('@[^\s]+',' ',text)
    text = " ".join(re.split(r'\s|-', text))
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'#covid', r' covid ', text)
    text = re.sub(r'#corona', r' corona ', text)
    text = re.sub(r'#coronavirus', r' coronavirus ', text)
    text = re.sub(r'#([^\s]+)', r' ', text)
    text = ''.join([i for i in text if not i.isdigit()])
    text = re.sub(r"(\!)\1+", ' ', text)
    text = re.sub(r"(\?)\1+", ' ', text)
    text = re.sub(r"(\.)\1+", ' ', text)
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', ' ', text)
    text = re.sub('&amp;', 'and', text)
    

    #stoplist = stopwords.words('english')
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not') 
    stop_words.discard('and')
    stop_words.discard('but')

    
    
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator) # Technique 7: remove punctuation


    tokens = nltk.word_tokenize(text)
    tokens = [tokens for tokens in tokens if not tokens in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]


    tagged = nltk.pos_tag(tokens) # Technique 13: part of speech tagging 
    allowedWordTypes = ["J","R","V","N"] #  J is Adjective, R is Adverb, V is Verb, N is Noun. These are used for POS Tagging
    final_text = []
    for w in tagged:

        if (w[1][0] in allowedWordTypes):
            final_word = addCapTag(w[0])
            final_word = lemmatizer.lemmatize(final_word)
            final_text.append(final_word)
            text = " ".join(final_text) 
    
    return text


dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

#print(len(dataset))

X=dataset['Cleaned Tweet Text']
Y=dataset['Result']

""" X_train = X[:49491]
X_test = X[49491:]

Y_train = Y[:49491]
Y_test = Y[49491:] """

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#print("Xtrain =",len(X_train))
#print("Xtest =",len(X_test))
#print("ytrain =",len(Y_train))
#print("ytest =",len(Y_test))
                            
tfidf_vectorizer = TfidfVectorizer()#ngram_range=(2,2)) #increases accuracy from 85 to 93 percent in case of SVM and a similar impact on other algos, but classification of new tweets is not proper and when csv file is finalD2.
x_train = tfidf_vectorizer.fit_transform(X_train) 
x_test = tfidf_vectorizer.transform(X_test)


""" BNBmodel = BernoulliNB()
BNBmodel.fit(x_train, Y_train)
                    
y_pred = BNBmodel.predict(x_test)

print("Naive Bayes accuracy_score is", accuracy_score(Y_test,y_pred)*100)
print("Naive Bayes precision_score is", precision_score(Y_test,y_pred, average='micro')*100)
print("Naive Bayes recall_score is", recall_score(Y_test,y_pred, average="micro")*100)
print("Naive Bayes f1_score is", f1_score(Y_test,y_pred, average="micro")*100)

 """
SVMmodel = LinearSVC(C=1)
SVMmodel.fit(x_train, Y_train)

y_predSVM = SVMmodel.predict(x_test)

data = pd.read_csv("finalD.csv", encoding_errors='ignore', low_memory=False)
data = data.dropna()
data = data.reset_index(drop=True)

print(len(data))

XX=data['Cleaned Tweet Text']
YY=data['Result']

""" XX_train = XX[:98982]
XX_test = XX[98982:]

YY_train = YY[:98982]
YY_test = YY[98982:] """

XX_train, XX_test, YY_train, YY_test = train_test_split(XX,YY,test_size=0.2,random_state=42)

print("Xtrain =",len(XX_train))
print("Xtest =",len(XX_test))

tfacc = TfidfVectorizer()#ngram_range=(2,2))
xx_train = tfacc.fit_transform(XX_train) 
xx_test = tfacc.transform(XX_test)

SVMaccModel = LinearSVC(C=1)
SVMaccModel.fit(xx_train, YY_train)
y_predaccSVM = SVMaccModel.predict(xx_test)

print("SVM accuracy_score is", accuracy_score(YY_test, y_predaccSVM)*100)
print("SVM precision_score is", precision_score(YY_test,y_predaccSVM, average="micro")*100)
print("SVM recall_score is", recall_score(YY_test, y_predaccSVM, average="micro")*100)
print("SVM f1_score is", f1_score(YY_test, y_predaccSVM, average="micro")*100)

""" 
When SVM was used for prediction.
61862
Xtrain== 49489
Xtest== 12373
SVM accuracy_score is 74.86462458579165
SVM precision_score is 74.86462458579165
SVM recall_score is 74.86462458579165
SVM f1_score is 74.86462458579165
Random Forest accuracy_score is 73.30477652954013
Random Forest precision_score is 73.30477652954013
Random Forest recall_score is 73.30477652954013
Random Forest f1_score is 73.30477652954013
Vader Result Count 61 436 4
ML Result count 91 409 1 

"""

""" RFModel = RandomForestClassifier(n_estimators = 100)
RFModel.fit(x_train, Y_train)

y_predRF = RFModel.predict(x_test)

print("Random Forest accuracy_score is", accuracy_score(Y_test, y_predRF)*100)
print("Random Forest precision_score is", precision_score(Y_test,y_predRF, average="micro")*100)
print("Random Forest recall_score is", recall_score(Y_test, y_predRF, average="micro")*100)
print("Random Forest f1_score is", f1_score(Y_test, y_predRF, average="micro")*100)

When Random Forest algo was used for prediction.

61862
Xtrain== 49489
Xtest== 12373 
SVM accuracy_score is 74.89695304291602
SVM precision_score is 74.89695304291602
SVM recall_score is 74.89695304291602
SVM f1_score is 74.89695304291602
Random Forest accuracy_score is 74.137234300493
Random Forest precision_score is 74.137234300493
Random Forest recall_score is 74.137234300493
Random Forest f1_score is 74.137234300493
Vader Result Count 61 436 4
ML Result count 23 477 1

 """
""" LGmodel = LogisticRegression(C=1, max_iter=1000)
LGmodel.fit(x_train, Y_train)

y_predLG = LGmodel.predict(x_test)

print("Logistic Regression accuracy_score is", accuracy_score(Y_test, y_predLG)*100)
print("Logistic Regression precision_score is", precision_score(Y_test,y_predLG, average="micro")*100)
print("Logistic Regression recall_score is", recall_score(Y_test, y_predLG, average="micro")*100)
print("Logistic Regression f1_score is", f1_score(Y_test, y_predLG, average="micro")*100)
 """
q = "lang:en(((India OR Pakistan) AND (covid AND economic crisis)) AND (income OR tax OR gdp OR corona OR pandemic OR covid crisis OR poor OR #COVID19 OR #Corona OR #Coronavirus))"# since:2019-12-05 until:2022-12-05)"

l = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(q).get_items()):
    if i > 500:
        break
    else:
        t = sentiment(cleaning(tweet.rawContent))
        if t > 0:
            l.append([tweet.id, tweet.rawContent, cleaning(tweet.rawContent), t, 'Positive'])
        elif t < 0:
            l.append([tweet.id, tweet.rawContent, cleaning(tweet.rawContent), t, 'Negative'])
        else:
            l.append([tweet.id, tweet.rawContent, cleaning(tweet.rawContent), t, 'Neutral'])

# Creating a dataframe from the tweets list above 
df = pd.DataFrame(l, columns=['Tweet ID', 'Text', 'Cleaned Tweet Text', 'Compound', 'Result'])

df.to_csv('compare.csv', index=False)

test_data = pd.read_csv("compare.csv")

print(len(test_data))

p = 0
n = 0
nn = 0

for i in test_data['Result']:
    if i == 'Positive':
        p += 1
    elif i == 'Negative':
        n += 1
    else:
        nn += 1

print("Vader Result\n\nPositive", p, "\nNegative", n, "\nNeutral", nn, "\n\n")

ml_pred = pd.read_csv("compare.csv")

ml_pred1 = tfidf_vectorizer.transform(ml_pred['Cleaned Tweet Text'])

test_pred = SVMmodel.predict(ml_pred1) # Using SVM here

dicti = {'Prediction': test_pred}

df1 = pd.DataFrame(next(iter(dicti.values())), columns=['ML Result'])

out = pd.merge(ml_pred, df1, left_index=True, right_index=True)

out.to_csv("compareres.csv", index=False)

ou = pd.read_csv('compareres.csv', encoding_errors='ignore')

p1 = 0
n1 = 0
nn1 = 0

for i in ou['ML Result']:
    if i == 'Positive':
        p1 += 1
    elif i == 'Negative':
        n1 += 1
    else:
        nn1 += 1

print("ML Result\n\nPositive", p1, "\nNegative", n1, "\nNeutral", nn1, "\n\n")

""" 
To check for overfitting of model

SVMmodel = LinearSVC(C=1)
SVMmodel.fit(x_train, Y_train)

y_predSVM = SVMmodel.predict(x_test)

y_predS = SVMmodel.predict(x_train)

print("SVM accuracy_score for test is", accuracy_score(Y_test, y_predSVM)*100)
print("SVM accuracy_score for train is", accuracy_score(Y_train, y_predS)*100)
"""

""" res = ''

for i in range(len(ou['Cleaned Tweet Text'])):
    res += ou['Cleaned Tweet Text'][i] + ' '

res_split = res.split()

cnt = Counter(res_split)

most = cnt.most_common(500)

r = ''

for i in most:
    r += i[0] + ' '

#print(r)

sid_obj = SentimentIntensityAnalyzer()
sentiment_dict = sid_obj.polarity_scores(r)

if sentiment_dict['compound'] >= 0.05 :
    print("Overall Sentiment is Positive")
elif sentiment_dict['compound'] <= - 0.05 :
    print("Overall Sentiment is Negative")
else:
    print("Overall Sentiment is Neutral") """

if p1 < n1:
    print("Overall sentiment is Negative")
elif p1 > n1:
    print("Overall sentiment is Positive")
else:
    print("Overall sentiment is Neutral")