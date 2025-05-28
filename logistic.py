from data import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

train_data, test_data = load_dataset("imdb_reviews")

vectorizer = CountVectorizer(stop_words="english")
X_tr = vectorizer.fit_transform(train_data[:, 0])
y_tr = train_data[:, 1]
X_te = vectorizer.fit_transform(test_data[:, 0])
y_te = test_data[:, 1]
print(X_tr)

model = LogisticRegression()
model.fit(X_tr, y_tr)
