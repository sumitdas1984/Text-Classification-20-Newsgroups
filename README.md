# Text-Classification-20-Newsgroups

The dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.

* Builded vocabulary from the dataset which was used as a feature set.
* Implemented Multinomial Naive Bayes classifier from scratch for classifying news into appropriate group.

Dataset : http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

## Results

Architecture1
------------------------------------------------------
model = Sequential()
# embedding layer
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(128, activation='relu'))
# output layer
model.add(layers.Dense(20, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
======================================================
epochs = 20

Training Accuracy: 0.9772
Testing Accuracy:  0.7914
======================================================

Architecture2
------------------------------------------------------
model = Sequential()
# embedding layer
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(128, activation='relu'))
# output layer
model.add(layers.Dense(20, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
======================================================
epochs = 20

Training Accuracy: 0.9766
Testing Accuracy:  0.7880
======================================================
