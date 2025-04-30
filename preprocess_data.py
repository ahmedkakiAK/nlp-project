import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

def remove_intro(text):
    # Remove everything from the beginning up to and including the first dash followed by a space
    return re.sub(r'^.*?-\s+', '', text)

def clean_and_merge(row):
    # Merge title and text
    full_text = f"{row['title']} {row['text']}"
    # Lowercase
    full_text = full_text.lower()
    # Remove special characters (keep letters, numbers, and spaces)
    full_text = re.sub(r'[^a-z0-9\s]', '', full_text)
    return full_text

df_true = pd.read_csv('data/isot/raw/True.csv')
df_fake = pd.read_csv('data/isot/raw/Fake.csv')
print('Data loaded successfully.')

df_true['text'] = df_true['text'].apply(remove_intro)
print('Intro removed from true data.')

df_true['label'] = 1
df_fake['label'] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)
print('Data concatenated with labels.')

df['full_text'] = df.apply(clean_and_merge, axis=1)
print('Full text column created.')

X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['label'], test_size=0.2, random_state=99)
print('Data split into training and testing sets.')

df_train = pd.DataFrame({'text': X_train, 'label': y_train})
df_test = pd.DataFrame({'text': X_test, 'label': y_test})
df_train.to_csv('data/isot/preprocessed/train.csv', index=False)
df_test.to_csv('data/isot/preprocessed/test.csv', index=False)
print('ISOT train and test datasets preprocessed and saved.')

print('------------------------')

df = pd.read_csv('data/kaggle/raw/fake_or_real_news.csv')
print('Kaggle data loaded successfully.')

df = df.drop(columns=['Unnamed: 0'])
df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})
print('Labels mapped to 0 and 1.')

df['full_text'] = df.apply(clean_and_merge, axis=1)
print('Full text column created.')

X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['label'], test_size=0.2, random_state=99)
print('Data split into training and testing sets.')

df_train = pd.DataFrame({'text': X_train, 'label': y_train})
df_test = pd.DataFrame({'text': X_test, 'label': y_test})
df_train.to_csv('data/kaggle/preprocessed/train.csv', index=False)
df_test.to_csv('data/kaggle/preprocessed/test.csv', index=False)
print('Kaggle train and test datasets preprocessed and saved.')



