
# coding: utf-8

# In[8]:


import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers


# In[2]:


# create input dataframe
def create_input_dataframe(dirpath):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(dirpath)):
        path = os.path.join(dirpath, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

    data = pd.DataFrame({'document': texts, 'label': labels})
    data = data.sample(frac=1)
    return data


# In[3]:


# create embedding matrix from pretrained embedding model
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding='utf8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


# In[4]:


data = create_input_dataframe('data/20_newsgroup/')
print('input dataframe shape: ' + str(data.shape))
data = data.sample(frac=1)
data.head()


# In[9]:


# get the documents and labels in numpy array
documents = data['document'].values
y = to_categorical(np.asarray(data['label'].values))


# In[10]:


documents_train, documents_test, y_train, y_test = train_test_split(documents, y, test_size=0.25, random_state=1000)


# In[11]:


MAX_NUM_WORDS = 20000
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer.fit_on_texts(documents_train)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print('vocab_size: ' + str(vocab_size))


# In[12]:


X_train = tokenizer.texts_to_sequences(documents_train)
X_test = tokenizer.texts_to_sequences(documents_test)

maxlen = 1000

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[13]:


embedding_dim = 50
embedding_matrix = create_embedding_matrix('resource/glove.6B.50d.txt', tokenizer.word_index, embedding_dim)


# In[14]:


nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
overlap = nonzero_elements / vocab_size
print('vocab overlap: ' + str(overlap))


# In[15]:


model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
# model.add(layers.Flatten())
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))


# In[16]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[59]:


history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=32)


loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# save the models and weight for future purposes
# serialize model to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk")
