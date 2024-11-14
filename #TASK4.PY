import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv(r'D:\Python\PRODIGY INFOTECH INTERNSHIP\TASK#4\twitter_training.csv')  # replace 'path_to_file.csv' with your dataset file path

# Display the first few rows
data.head()
# Check dataset information
data.info()
print(data.columns)
# Check for missing values
print(data.isnull().sum())

data = data.rename(columns={'Positive': 'sentiment'})

# Check the unique values in the sentiment column
print(data['sentiment'].unique())

data = data.rename(columns={'im getting on borderlands and i will murder you all ,':'text'})
# Example of preprocessing (lowercase and removing non-alphabetic characters)
data['cleaned_text'] = data['text'].str.lower().str.replace('[^a-z\s]', '', regex=True)

print(data[['text','cleaned_text','sentiment']].head())

# Optional: Tokenize and remove stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join(
   # word for word in x.split() if word not in stop_words
#))
# Plot the sentiment distribution
sns.countplot(x='sentiment', data=data, palette='viridis')
plt.title('Sentiment Distribution')
plt.show()

# WordCloud for positive sentiment
positive_text = ' '.join(data[data['sentiment'] == 'positive']['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Sentiment')
plt.show()

# Repeat the above for 'negative' and 'neutral' sentiments
# Example for positive sentiment word frequency
from collections import Counter

positive_words = ' '.join(data[data['sentiment'] == 'positive']['cleaned_text']).split()
positive_word_counts = Counter(positive_words)
positive_word_df = pd.DataFrame(positive_word_counts.most_common(20), columns=['word', 'count'])

sns.barplot(x='count', y='word', data=positive_word_df, palette='viridis')
plt.title('Top 20 Words in Positive Sentiment')
plt.show()

# Convert to datetime if necessary
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Plot sentiment over time
data.set_index('timestamp').groupby(['sentiment']).resample('M').size().unstack(0).plot(kind='line', figsize=(12, 6))
plt.title('Sentiment Trends Over Time')
plt.ylabel('Number of Tweets')
plt.xlabel('Time')
plt.show()