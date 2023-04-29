import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import timeit

start = timeit.default_timer()

# All the program statements
# Load data
data = []
for label in ['spam', 'ham']:
    for file in os.listdir(label):
        with open(os.path.join(label, file), 'r', encoding="ascii") as f:
            try:
                email = f.read()
            except:
                pass
        data.append((label, email))
data = pd.DataFrame(data, columns=['label', 'email'])


data['email'] = data['email'].str.lower().str.replace('[^\w\s]', '', regex=True)
# replacing any consecutive white spaces within the email address with a single space
data['email'] = data['email'].str.split().apply(lambda x: ' '.join(x))


# Feature extraction
# Convert text to token counts, excluding common English words
vectorizer = CountVectorizer(stop_words='english')
# Transform the input data into a matrix of token counts
X_counts = vectorizer.fit_transform(data['email'])

tfidf_transformer = TfidfTransformer()
# Compute TF-IDF (term frequency - inverse document frequency) matrix
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Split TF-IDF matrix and spam/ham into training and testing set
# X_train/y_train (X_test/y_test) contains training (testing) set of features/labels

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.2, random_state=42)

stop1 = timeit.default_timer()
execution_time1 = stop1 - start
print("Starting SVC")
# Model training
# Linear SVC
svm = SVC(kernel='linear', C=1.0, probability=True)
# Train with the training set of features and labels
svm.fit(X_train, y_train)
# Use trained svm to predict label of X_test
y_pred = svm.predict(X_test)

# fraction of correctly classified data points
accuracy = accuracy_score(y_test, y_pred)
# fraction of true positives among classified as positive
precision = precision_score(y_test, y_pred, pos_label='spam')
# proportion of actual positives was identified correctly
recall = recall_score(y_test, y_pred, pos_label='spam')
# harmonic mean of precision and recall
f1 = f1_score(y_test, y_pred, pos_label='spam')
print(f"SVM: Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

stop2 = timeit.default_timer()
execution_time2 = stop2 - stop1
print(f"End of SVC, time elapsed : {execution_time2:.2f}")

new_email = ['The Mathematics and Statistics Department is currently hiring part-time students who are paid $450 weekly. The Tasks given are very simple and are being performed remotely. This position is open to all students of UNIVERSITE DE MONTREAL regardless of their departments. For more information, please submit your full name, valid mobile phone number, year, year of study and department via this email or contact Professor Frigon Marlene (774) 460-1833 to receive the job description or application requirements. Sincerely, Professor Frigon Marlene Department of Mathematics and Statistics.']
new_email_counts = vectorizer.transform(new_email)
new_email_tfidf = tfidf_transformer.transform(new_email_counts)

stop3 = timeit.default_timer()
print("Start of prediction of new email using SVC")

new_email_pred = svm.predict(new_email_tfidf)
print(new_email_pred)

stop4 = timeit.default_timer()
execution_time3 = stop4 - stop3
print(f"End of prediction of new email using SVC, time elapsed : {execution_time3:.2f}")
print("Start of Naive Bayes")

# Model training
# Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Use trained NB model to predict label of X_test
y_pred = clf.predict(X_test)
# Evaluate the performance of the classifier
accuracynb = accuracy_score(y_test, y_pred)
precisionnb = precision_score(y_test, y_pred, pos_label='spam')
recallnb = recall_score(y_test, y_pred, pos_label='spam')
f1nb = f1_score(y_test, y_pred, pos_label='spam')
print(f"Naive Bayes: Accuracy: {accuracynb:.2f}, Precision: {precisionnb:.2f}, Recall: {recallnb:.2f}, F1-Score: {f1nb:.2f}")

stop5 = timeit.default_timer()
execution_time4 = stop5 - stop4
print(f"End of NB, time elapsed : {execution_time4:.2f}")
print("Start of prediction of new email using NB")

new_email_pred_nb = clf.predict(new_email_tfidf)
print(new_email_pred_nb)

stop6 = timeit.default_timer()
execution_time5 = stop6 - stop5
print(f"End of prediction using NB, time elapsed : {execution_time5:.2f}")