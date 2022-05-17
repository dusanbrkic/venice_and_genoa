import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def skloni_interpunkciju(recenica):
    for ch in string.punctuation:
        recenica = recenica.replace(ch, '')
    return recenica


def skloni_stopwords(recenica):
    for stop in text.ENGLISH_STOP_WORDS:
        recenica = recenica.replace(stop.lower(), '')
    return recenica


def main():
    # print(sys.argv)
    #train_file_path = sys.argv[1]
    #test_file_path = sys.argv[2]

    train_file_path = "res/train.json"
    test_file_path = "res/test.json"

    train_data = pd.read_json(train_file_path)
    test_data = pd.read_json(test_file_path)

    train_data['text'] = train_data['text'].apply(
        lambda x: skloni_interpunkciju(x))
    train_data['text'] = train_data['text'].apply(lambda x: x.lower())
    train_data['text'] = train_data['text'].apply(
        lambda x: skloni_stopwords(x))

    test_data['text'] = test_data['text'].apply(
        lambda x: skloni_interpunkciju(x))
    test_data['text'] = test_data['text'].apply(lambda x: x.lower())
    test_data['text'] = test_data['text'].apply(
        lambda x: skloni_stopwords(x))

    X_train = train_data['text']
    Y_train = train_data['clickbait']

    X_test = test_data['text']
    Y_test = test_data['clickbait']

    vec = CountVectorizer()
    # print(X_train)
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)
    # print(vec)

    klasifikator = svm.SVC(kernel='rbf', gamma=0.15)
    klasifikator.fit(X_train, Y_train)

    y_predicted = klasifikator.predict(X_test)

    print(classification_report(Y_test, y_predicted))
    print(f1_score(Y_test, y_predicted))


if __name__ == '__main__':
    main()
