import sys
import pandas as pd
import string
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

def skloni_interpunkciju(recenica):
    for ch in string.punctuation:
        recenica = recenica.replace(ch, '')
    return recenica


def main():
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    train_data = pd.read_json(train_file_path)
    test_data = pd.read_json(test_file_path)

    train_data['text'] = train_data['text'].apply(
        lambda x: skloni_interpunkciju(x))
    
    test_data['text'] = test_data['text'].apply(
        lambda x: skloni_interpunkciju(x))

    X_train = train_data['text']
    Y_train = train_data['clickbait']

    X_test = test_data['text']
    Y_test = test_data['clickbait']

    vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, binary=True, min_df=2, max_df=0.5, lowercase=False)

    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)

    klasifikator = svm.SVC(kernel='sigmoid', gamma=1)
    klasifikator.fit(X_train, Y_train)

    y_predicted = klasifikator.predict(X_test)

    # print(classification_report(Y_test, y_predicted))
    print(f1_score(Y_test, y_predicted, average='micro'))


if __name__ == '__main__':
    main()
