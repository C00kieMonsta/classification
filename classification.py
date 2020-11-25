import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

df = pd.read_csv('labeled_description.csv', sep=',')
print(df.head())


# Lets extract descriptions (X) and categories (Y)
X = df['Description']
y = df['Category']

# Split data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Object to convert training set into a vector of tokens and normalize all vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set

# we use Support Vector Classifier to train our model
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)
text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC()),])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf.predict(X_test)

# How well is the model
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))

# Test your model
output_categories = text_clf.predict(['Dogs are awesome', 'I do not want cats in my house'])

print(output_categories)


