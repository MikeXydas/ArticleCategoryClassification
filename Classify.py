
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import STOPWORDS
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn import preprocessing
from sklearn.model_selection import KFold
from gensim.parsing.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from nltk.tokenize.moses import MosesDetokenizer

import string
import pandas as pd
import nltk

train_data = pd.read_csv('train_set.csv', sep="\t", encoding="utf-8")

#train_data = train_data[0: 500] #Usinga part of the train_set for fast testing

#preprocessing of texts
myStopWord = set(ENGLISH_STOP_WORDS).union(STOPWORDS).union(stopwords.words('english'))
printable = set(string.printable)
stm = PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lematizer = nltk.stem.WordNetLemmatizer()
detokenizer = MosesDetokenizer()


def lem_text(text):
    return [lematizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


#Aims to have a percentage of a the text as the title
def apply_title(titleOfText, contentOfText):
    tLength = float(len(titleOfText.split()))
    cLength = float(len(contentOfText.split()))
    k = (0.065 * cLength) / tLength     #this value can be tweaked for optimised accuracy
    k = int(round(k))
    contentOfText += " " + " ".join([titleOfText for i in range(k)])
    return contentOfText


Texts = list()

for whichText, whichTitle in zip(train_data["Content"], train_data["Title"]):
    whichText = whichText.encode("ascii", errors="ignore")       # removing utf-8 encodings
    whichTitle = whichTitle.encode("ascii", errors="ignore")
    whichText = apply_title(whichTitle, whichText)
    whichText = whichText.lower()                                                           # turn all words lowercase
    whichText = whichText.translate(string.maketrans("", ""), string.punctuation)           # removing punctuation
    whichText = ' '.join([word for word in whichText.split() if word not in myStopWord])    # removing stop_words
    whichText = lem_text(whichText)                                                         # lemmatization
    whichText = " ".join(whichText)
    #whichText = stm.stem_sentence(whichText)                                                #stemming was removed as it resulted in a drop of accuracy
    Texts.append(whichText)

#label encoding categories and creating a set of them
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
setLabeledCats = set(y)

print "       --->     Testing of different max document frequencies without LSI on 12000 points    <---"
print "\n"

for i in [0.6, 0.7, 0.8, 0.85, 0.9]:        #We tweak the max_df of the Tfidvectorizer for optimum accuracy
    print "Max document frequency = ", i

    #vectorizing
    vectorizer = TfidfVectorizer(max_df=i, min_df=0.01, strip_accents='unicode', analyzer='word')
    X = vectorizer.fit_transform(Texts)

    svd = TruncatedSVD(n_components=107, n_iter=7, random_state=42)

    X_trunc = svd.fit_transform(X)

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
                oob_score=False, random_state=10, verbose=0, warm_start=False)

    gnb = MultinomialNB()

    svc = SVC(C=1000)

    neigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=7, p=2,
               weights='uniform')

    kf = KFold(n_splits=10, random_state=None, shuffle=False)

    def performClf(whichClf):
        accTot = 0
        if whichClf == gnb:
            X_gnb = X.toarray()
            for train_index, test_index in kf.split(X_gnb):
                X_train, X_test = X_gnb[train_index], X_gnb[test_index]
                y_train, y_test = y[train_index], y[test_index]
                whichClf.fit(X_train, y_train)
                y_predicted = whichClf.predict(X_test)
                accTot += accuracy_score(y_test, y_predicted)
        else:
            for train_index, test_index in kf.split(X_trunc):
                X_train, X_test = X_trunc[train_index], X_trunc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                whichClf.fit(X_train, y_train)
                y_predicted = whichClf.predict(X_test)
                accTot += accuracy_score(y_test, y_predicted)
        return accTot/10

    for whichClf in ["Random Forest", "SVM", "KNN", "Naive Bayes"]:
        if whichClf == "Naive Bayes":
            acc = performClf(gnb)
        elif whichClf == "Random Forest":
            acc = performClf(clf)
        elif whichClf == "SVM":
            acc = performClf(svc)
        else:
            acc = performClf(neigh)
        print "    Algorithm used: ", whichClf, "  | Mean accuracy: ", acc
    print "\n"




