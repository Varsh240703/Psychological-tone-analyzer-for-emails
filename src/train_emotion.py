import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/eng_dataset.csv")

# Preprocessing
df['content'] = df['content'].astype(str).str.lower()

# Check class distribution
class_counts = df['sentiment'].value_counts()
print("Class distribution before undersampling:\n", class_counts)

# Find the minimum size of the minority classes
min_class_size = class_counts.min()

# Undersample the 'fear' class
fear_class = df[df['sentiment'] == 'fear']
non_fear_classes = df[df['sentiment'] != 'fear']

# Randomly downsample the fear class
fear_class_undersampled = fear_class.sample(n=min_class_size, random_state=42)

# Combine the undersampled 'fear' class with the non-fear classes
df_balanced = pd.concat([fear_class_undersampled, non_fear_classes])

# Check the class distribution after undersampling
class_counts_balanced = df_balanced['sentiment'].value_counts()
print("\nClass distribution after undersampling:\n", class_counts_balanced)

# Features and labels
X = df_balanced['content']
y = df_balanced['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer (including bigrams) and Logistic Regression with class weight
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),  # Using bigrams
    ('clf', LogisticRegression(max_iter=1000))  # No class weights needed after balancing
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nEmotion Detection Model Trained Successfully")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save the pipeline
with open('models/emotion/emotion_pipeline.pkl', 'wb') as pipeline_file:
    pickle.dump(pipeline, pipeline_file)

print("\nPipeline saved successfully.")
