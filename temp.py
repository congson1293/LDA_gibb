import numpy as np
import lda
import lda.datasets
from scipy.sparse import csr_matrix



X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()

model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)

model.fit(X)

topic_word = model.topic_word_

n_top_words = 10

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

new_doc = np.zeros(shape=(len(vocab)), dtype=np.int)
new_doc[0] = 1
new_doc[100] = 2
new_doc[32] = 1
print model.transform(new_doc)