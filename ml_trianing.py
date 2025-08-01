import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("packet_data.csv")

# Example features — adjust based on your dataset
features = ['priority_tag', 'eth_type', 'dscp', 'src', 'dst', 'protocol', 'sport', 'dport']
target = 'priority_class'  # this should be the class label you want to predict

# If needed, encode IPs and categorical fields
df['src'] = df['src'].astype('category').cat.codes
df['dst'] = df['dst'].astype('category').cat.codes

X = df[features]
y = df[target]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "ml_priority_model.joblib")
print("Model saved to ml_priority_model.joblib")
