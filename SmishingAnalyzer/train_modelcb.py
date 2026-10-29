import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report



#initializes data 

data = pd.read_csv('tuffData.csv', encoding='latin-1', names=['text', 'label'], on_bad_lines='skip')

testdata = pd.read_csv('text.csv', encoding='latin-1', names=['text'], on_bad_lines='skip')

if data.columns[0] == 'text' and data.columns[1] == 'label':
    data = data.rename(columns={'text': 'text', 'label': 'label'})
else:
    data.columns = ['text', 'label']

data = data.dropna(subset=['label', 'text'])

data['label'] = data['label'].str.strip().map({'0':0, '1':1})

data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)

print("Duplicate texts:", data['text'].duplicated().sum())
#removes duplicates
data = data.drop_duplicates(subset=['text'])

print("Duplicate texts:", data['text'].duplicated().sum())

X = data['text']
y = data['label']

#split train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

overlap = set(X_train) & set(X_test)

print("Exact text overlap:", len(overlap))

print("Shape of the training set:", X_train.shape, y_train.shape)
print("Shape of the testing set:", X_test.shape, y_test.shape)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=100)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

new_pred = model.predict(vectorizer.transform((testdata['text'])))

# Predict text.csv
test_tfidf = vectorizer.transform(testdata['text'])

new_pred = model.predict(test_tfidf).flatten()
probabilities = model.predict_proba(test_tfidf)

#print results

print("Total rows:", len(data))

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')


#runs through test data

for i, (text, pred, probs) in enumerate(
    zip(testdata['text'], new_pred, probabilities)
):
    if pred == 1:
        classification = "smishing"
        confidence = probs[1]
    else:
        classification = "safe"
        confidence = probs[0]

    print(f"\nEmail {i + 1}")
    print(f"Text: {text}")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2%}")
