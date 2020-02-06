# https://github.com/lda-project/lda

import os
import utils
import data_loader
import lda
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import config
from io import open
import numpy as np
import preprocessing



class lda_feature_extraction:
    def __init__(self, ntopics, root_dir='.'):
        self.ntopics = ntopics
        self.vectorizer = None
        self.model = None
        self.root_dir = root_dir


    def load(self, model):
        print('loading %s ...' % (model))
        if os.path.isfile(model):
            return joblib.load(model)
        else:
            return None


    def load_model(self):
        print('load model ...')
        vectorizer = self.load(os.path.join(self.root_dir, 'model/vectorizer.pkl'))
        model = self.load(os.path.join(self.root_dir, 'model/model.pkl'))
        return model, vectorizer


    def save(self, model, path):
        print('saving %s ...' % (path))
        utils.mkdir('model')
        joblib.dump(model, path, compress=True)
        return


    def save_model(self):
        utils.mkdir(os.path.join(self.root_dir, 'model'))
        self.save(self.vectorizer, os.path.join(self.root_dir, 'model/vectorizer.pkl'))
        self.save(self.model, os.path.join(self.root_dir, 'model/model.pkl'))


    def show_topics(self, vocab, n_top_words, output_file):
        with open(output_file, 'w', encoding='utf-8') as fp:
            data = []
            for i, topic_dist in enumerate(self.model.topic_word_):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
                topic_data = u'Topic {}: {}'.format(i, u' '.join(topic_words))
                print(topic_data)
                data.append(topic_data)
            fp.write(u'\n'.join(data))


    def train(self):
        print('training ...')
        documents = data_loader.load_data(config.DATASET)

        stopwords, _ = utils.load_data2list_string(os.path.join(self.root_dir, 'stopwords.txt'))
        self.vectorizer = CountVectorizer(max_df=0.7, min_df=2,
                                          stop_words=stopwords,
                                          max_features=5000)

        doc_vector = self.vectorizer.fit_transform(documents)
        vocab = self.vectorizer.get_feature_names()

        self.model = lda.LDA(n_topics=self.ntopics, n_iter=1000, random_state=1)

        print('fit model ...')
        self.model.fit(doc_vector)

        utils.mkdir('model')
        self.show_topics(vocab, 20, 'model/topics.txt')

        self.save_model()


    def run(self):
        model, vectorizer = self.load_model()
        if model is None or vectorizer is None:
            self.train()
        else:
            self.model = model
            self.vectorizer = vectorizer


    # docs is a list of document
    def infer_doc_topic(self, docs):
        docs = map(lambda d: preprocessing.preprocessing(d), docs)
        docs_vector = self.vectorizer.transform(docs)
        return self.model.transform(docs_vector)




if __name__ == '__main__':
    l = lda_feature_extraction(ntopics=300)
    l.run()

    with open('TT_NLD_ (4699).txt', 'r', encoding='utf-16') as fp:
        content = fp.read()
        doc_topic = l.infer_doc_topic([content])
        pass