"""Implement the Multinomial Naïve Bayes text classification with La-Place smoothing and show the predicted class for the given new sentence. Use the following dataset.Code in python 

Training Data:
S.No.  Document                            Class. 
1.        Language Model Learning.   0
2.        Text Classification Model.    0
3.         Ngram Language Model.      0
4.       Natural Language Processing Task 0
5.       Image Processing Model.       1
6.       Computer Vision Task.            1
7.       Image Classification Model.   1
8.       Image Segmentation.              1
9.       Image Processing.                  1
10.     Object Recognition.                1

Test Data 
1.      Image Learning Task.             ?
2.      Text Learning Task.                ?
"""
from collections import defaultdict
import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha  # Laplace smoothing parameter
        self.vocab = set()
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_doc_counts = defaultdict(int)
        self.class_prior = defaultdict(float)

    def fit(self, X_train, y_train):
        for i in range(len(X_train)):
            document = X_train[i]
            label = y_train[i]
            self.class_doc_counts[label] += 1
            for word in document.split():
                self.class_word_counts[label][word] += 1
                self.vocab.add(word)

        total_documents = sum(self.class_doc_counts.values())
        for label in self.class_doc_counts:
            self.class_prior[label] = self.class_doc_counts[label] / total_documents

    def predict(self, X_test):
        predictions = []
        for document in X_test:
            scores = {}
            for label in self.class_prior:
                scores[label] = np.log(self.class_prior[label])
                for word in document.split():
                    # Calculate the log probability of each word given the class using Laplace smoothing
                    prob = (self.class_word_counts[label][word] + self.alpha) / (sum(self.class_word_counts[label].values()) + self.alpha * len(self.vocab))
                    scores[label] += np.log(prob)

            predicted_label = max(scores, key=scores.get)
            predictions.append(predicted_label)
        return predictions

def main():
    # Training data
    X_train = [
        "Language Model Learning",
        "Text Classification Model",
        "Ngram Language Model",
        "Natural Language Processing Task",
        "Image Processing Model",
        "Computer Vision Task",
        "Image Classification Model",
        "Image Segmentation",
        "Image Processing",
        "Object Recognition"
    ]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    # Test data
    X_test = [
        "Image Learning Task",
        "Text Learning Task"
    ]

    # Initialize and train the classifier
    clf = MultinomialNaiveBayes(alpha=1)
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Display the predicted classes for the test data
    print("Predicted classes for test data:")
    for i in range(len(X_test)):
        print(f"{X_test[i]} : Class {predictions[i]}")

if __name__ == "__main__":
    main()
