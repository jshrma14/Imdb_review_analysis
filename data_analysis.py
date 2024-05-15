import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_dataset(file_path):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(file_path)

def analyze_data(df):
    """
    Perform data analysis and visualization.
    """
    # Basic statistics
    print("Dataset Statistics:")
    print(df.describe())

    # Visualize review length distribution
    df['Review Length'] = df['Review'].apply(len)
    plt.hist(df['Review Length'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Review Length Distribution')
    plt.xlabel('Review Length')
    plt.ylabel('Frequency')
    plt.show()

    # Word cloud of most frequent words in reviews
    text = ' '.join(df['Review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Reviews')
    plt.show()

def prepare_data(df):
    """
    Prepare data for sentiment analysis.
    """
    # Convert reviews to numerical representation using bag-of-words
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Review'])

    # Create labels (positive/negative sentiment)
    y = (df['Title'].str.contains('The Godfather')).astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a random forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the trained model.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    # Load dataset
    df = load_dataset('dataset.csv')

    # Data analysis and visualization
    analyze_data(df)

    # Prepare data for sentiment analysis
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train model
    clf = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(clf, X_test, y_test)
