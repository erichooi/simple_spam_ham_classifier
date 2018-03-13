from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from argparse import ArgumentParser

import json
import pickle
import numpy as np

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--act", help="""
        save   need to provide train file and model
        evaluate   need to provide test file and model
        interact   need to provide model
        """)
    parser.add_argument("--train-file")
    parser.add_argument("--test-file")
    parser.add_argument("--model")
    return parser

def save_model(train_questions, train_labels, model):
    stopword = stopwords.words("english")
    text_clf = Pipeline([
        ("vect", CountVectorizer(
            stop_words=stopword,
        )),
        # ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB()),
    ])

    text_clf.fit(train_questions, train_labels)

    with open(model, "wb") as f:
        pickle.dump(text_clf, f)

def load_model(model):
    with open(model, "rb") as f:
        text_clf = pickle.load(f)
    return text_clf

def test_model(test_questions, test_labels, model):
    text_clf = load_model(model)
    predicted = text_clf.predict(test_questions)
    return (np.mean(predicted == test_labels))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print("SPAM FILTERING SYSTEM")
    if args.act == "save":
        train_file = args.train_file
        train_questions = []
        train_labels = []
        with open(train_file, "r") as f:
            for line in f:
                data = json.loads(line)
                question = "{}".format(data["Title"])
                train_questions.append(question)
                if data["Tag"] == "spam":
                    train_labels.append(1)
                else:
                    train_labels.append(0)
        save_model(train_questions, train_labels, args.model)
    elif args.act == "evaluate":
        test_file = args.test_file
        test_questions = []
        test_labels = []
        with open(test_file, "r") as f:
            for line in f:
                data = json.loads(line)
                question = "{}".format(data["Title"])
                test_questions.append(question)
                if data["Tag"] == "spam":
                    test_labels.append(1)
                else:
                    test_labels.append(0)
        result = test_model(test_questions, test_labels, args.model)
        print(result)
    elif args.act == "interact":
        text_clf = load_model(args.model)
        while True:
            print("Type your questions, or type \"exit\" to quit.")
            user_input = input("> ")
            print()
            if user_input == "exit":
                exit()
            else:
                predicted = text_clf.predict([user_input])
                if predicted == 1:
                    print("label: ", "spam")
                elif predicted == 0:
                    print("label: ", "ham")
            print()
