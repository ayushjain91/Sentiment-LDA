"""
Implementation of the collapsed Gibbs sampler for Sentiment-LDA, described in
Sentiment Analysis with Global Topics and Local Dependency (Li, Huang and Zhu)
"""

import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize,sent_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
st = PorterStemmer()


MAX_VOCAB_SIZE = 50000


def sampleFromDirichlet(alpha):
    """
    Sample from a Dirichlet distribution
    alpha: Dirichlet distribution parameter (of length d)
    Returns:
    x: Vector (of length d) sampled from dirichlet distribution

    """
    return np.random.dirichlet(alpha)


def sampleFromCategorical(theta):
    """
    Samples from a categorical/multinoulli distribution
    theta: parameter (of length d)
    Returns:
    x: index ind (0 <= ind < d) based on probabilities in theta
    """
    theta = theta/np.sum(theta)
    return np.random.multinomial(1, theta).argmax()


def word_indices(wordOccuranceVec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in wordOccuranceVec.nonzero()[0]:
        for i in range(int(wordOccuranceVec[idx])):
            yield idx


class SentimentLDAGibbsSampler:

    def __init__(self, numTopics, alpha, beta, gamma, numSentiments=2):
        """
        numTopics: Number of topics in the model
        numSentiments: Number of sentiments (default 2)
        alpha: Hyperparameter for Dirichlet prior on topic distribution
        per document
        beta: Hyperparameter for Dirichlet prior on vocabulary distribution
        per (topic, sentiment) pair
        gamma:Hyperparameter for Dirichlet prior on sentiment distribution
        per (document, topic) pair
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numTopics = numTopics
        self.numSentiments = numSentiments

    def processSingleReview(self, review, d=None):
        """
        Convert a raw review to a string of words
        """
        letters_only = re.sub("[^a-zA-Z]", " ", review)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [st.stem(w) for w in words if w not in stops]
        return(" ".join(meaningful_words))
        # review = review.decode("ascii", "ignore").encode("ascii")
            
        # stops = set(stopwords.words("english"))
        # meaningful_words = []
        # sentences = sent_tokenize(review)
        # i = 0
        # for sentence in sentences:
        #     pos_words = pos_tag(word_tokenize(sentence))
        #     for (word, pos) in pos_words:
        #         word = word.lower()
        #         word = re.sub("[^a-zA-Z]", "", word)
        #         if word == "":
        #             continue
        #         if word.lower() in stops:
        #             continue
        #         wordnetpos = None
        #         if pos.startswith('N'):
        #             wordnetpos = 'n'
        #         elif pos.startswith('V'):
        #             wordnetpos = 'v'
        #         elif pos.startswith('J'):
        #             wordnetpos = 'a'
        #         elif pos.startswith('R'):
        #             wordnetpos = 'r'
        #         synsets = swn.senti_synsets(word, wordnetpos)
        #         posScore = np.mean([s.pos_score() for s in synsets])
        #         negScore = np.mean([s.neg_score() for s in synsets])
        #         if d is not None and posScore >= 0.1 and posScore > negScore:
        #             self.priorSentiment[(d, i)] = 1
        #         elif d is not None and negScore >= 0.1 and negScore > posScore:
        #             self.priorSentiment[(d, i)] = 0
        #         meaningful_words.append(st.stem(word))
        #         i += 1
        # return(" ".join(meaningful_words))

    def processReviews(self, reviews, saveAs=None, saveOverride=False):
        import os
        import dill
        if not saveOverride and saveAs and os.path.isfile(saveAs):
            [wordOccurenceMatrix, self.vectorizer] = dill.load(open(saveAs,'r'))
            return wordOccurenceMatrix
        processed_reviews = []
        i = 0
        for review in reviews:
            if((i + 1) % 1000 == 0):
                print "Review %d of %d" % (i + 1, len(reviews))
            processed_reviews.append(self.processSingleReview(review, i))
            i += 1
        self.vectorizer = CountVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          stop_words="english",
                                          max_features=MAX_VOCAB_SIZE)
        train_data_features = self.vectorizer.fit_transform(processed_reviews)
        wordOccurenceMatrix = train_data_features.toarray()
        if saveAs:
            dill.dump([wordOccurenceMatrix, self.vectorizer], open(saveAs, 'w'))
        return wordOccurenceMatrix

    def _initialize_(self, reviews, saveAs=None, saveOverride=False):
        """
        wordOccuranceMatrix: numDocs x vocabSize matrix encoding the
        bag of words representation of each document
        """
        self.wordOccuranceMatrix = self.processReviews(reviews, saveAs, saveOverride)
        numDocs, vocabSize = self.wordOccuranceMatrix.shape

        # Pseudocounts
        self.n_dt = np.zeros((numDocs, self.numTopics))
        self.n_dts = np.zeros((numDocs, self.numTopics, self.numSentiments))
        self.n_d = np.zeros((numDocs))
        self.n_vts = np.zeros((vocabSize, self.numTopics, self.numSentiments))
        self.n_ts = np.zeros((self.numTopics, self.numSentiments))
        self.topics = {}
        self.sentiments = {}
        self.priorSentiment = {}

        alphaVec = self.alpha * np.ones(self.numTopics)
        gammaVec = self.gamma * np.ones(self.numSentiments)

        for i, word in enumerate(self.vectorizer.get_feature_names()):
            synsets = swn.senti_synsets(word)
            posScore = np.mean([s.pos_score() for s in synsets])
            negScore = np.mean([s.neg_score() for s in synsets])
            if posScore >= 0.1 and posScore > negScore:
                self.priorSentiment[i] = 1
            elif negScore >= 0.1 and negScore > posScore:
                self.priorSentiment[i] = 0

        for d in range(numDocs):

            topicDistribution = sampleFromDirichlet(alphaVec)
            sentimentDistribution = np.zeros(
                (self.numTopics, self.numSentiments))
            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)
            for i, w in enumerate(word_indices(self.wordOccuranceMatrix[d, :])):
                t = sampleFromCategorical(topicDistribution)
                s = sampleFromCategorical(sentimentDistribution[t, :])

                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_dts[d, t, s] += 1
                self.n_d[d] += 1
                self.n_vts[w, t, s] += 1
                self.n_ts[t, s] += 1



    def conditionalDistribution(self, d, v):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))
        firstFactor = (self.n_dt[d] + self.alpha) / \
            (self.n_d[d] + self.numTopics * self.alpha)
        secondFactor = (self.n_dts[d, :, :] + self.gamma) / \
            (self.n_dt[d, :] + self.numSentiments * self.gamma)[:, np.newaxis]
        thirdFactor = (self.n_vts[v, :, :] + self.beta) / \
            (self.n_ts + self.n_vts.shape[0] * self.beta)
        probabilities_ts *= firstFactor[:, np.newaxis]
        probabilities_ts *= secondFactor * thirdFactor
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

    def getTopKWordsByLikelihood(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(t, s | v) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (1, 2))
        pseudocounts /= normalizer[:, np.newaxis, np.newaxis]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                vocab = self.vectorizer.get_feature_names()
                print t, s, [vocab[i] for i in topWordIndices]

    def getTopKWords(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(v | t, s) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :, :]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                vocab = self.vectorizer.get_feature_names()
                print t, s, [vocab[i] for i in topWordIndices]


    def run(self, reviews, maxIters=30, saveAs=None, saveOverride=False):
        """
        Runs Gibbs sampler for sentiment-LDA
        """
        self._initialize_(reviews, saveAs, saveOverride)
        numDocs, vocabSize = self.wordOccuranceMatrix.shape
        for iteration in range(maxIters):
            print "Starting iteration %d of %d" % (iteration + 1, maxIters)
            for d in range(numDocs):
                for i, v in enumerate(word_indices(self.wordOccuranceMatrix[d, :])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1

                    probabilities_ts = self.conditionalDistribution(d, v)
                    if v in self.priorSentiment:
                        s = self.priorSentiment[v]
                        t = sampleFromCategorical(probabilities_ts[:, s])
                    else:
                        ind = sampleFromCategorical(probabilities_ts.flatten())
                        t, s = np.unravel_index(ind, probabilities_ts.shape)

                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1
