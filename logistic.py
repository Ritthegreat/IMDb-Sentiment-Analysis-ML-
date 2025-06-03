from data import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

train_data, test_data = load_dataset("imdb_reviews", cache=True)
vectorizer = TfidfVectorizer(stop_words="english")
X_tr = vectorizer.fit_transform(train_data[:, 0])
y_tr = train_data[:, 1]
X_te = vectorizer.transform(test_data[:, 0])
y_te = test_data[:, 1]

model = LogisticRegression(solver="liblinear", penalty="l2", C=1)
model.fit(X_tr, y_tr)
y_pred_tr = model.predict(X_tr)
y_pred = model.predict(X_te)
tr_acc = accuracy_score(y_pred_tr, y_tr)
te_acc = accuracy_score(y_pred, y_te)
print(f"Train Acc: {tr_acc}")
print(f"Test Acc: {te_acc}")
