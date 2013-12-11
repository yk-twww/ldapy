# -*- coding: utf-8 -*-

from random import randint, random
from operator import itemgetter


class ldapy(object):
    def set(self, docs, topics, alpha = 1.0, beta = 1.0):
        self.docs = docs
        self.topic_num = topics
        self.alpha = alpha
        self.beta = beta

        self._count_doc_word()


    def _count_doc_word(self):
        self.doc_num = len(self.docs)

        words = {}
        for doc in self.docs:
            for w in doc:
                words[w] = 0
        self.words = words.keys()


    def estimate(self, iter_num):
        self._init_state()

        for i in range(iter_num):
            self._gibbs()

        phi = self._init_zero_lls(len(self.words), self.topic_num)
        theta = self._init_zero_lls(self.topic_num, len(self.docs))
        for k in range(self.topic_num):
            total = sum(self.n_kt[k])
            for t in range(len(self.words)):
                phi[k][t] = (self.n_kt[k][t] + self.beta) / (total + self.beta * len(self.words))

        for m in range(len(self.docs)):
            total = sum(self.n_mk[m])
            for k in range(self.topic_num):
                theta[m][k] = (self.n_mk[m][k] + self.alpha) / (total + self.alpha * self.topic_num)

        return lda_res(len(self.docs), len(self.words), self.topic_num, phi, theta, self.words[:])

        
    def _init_state(self):
        self.state = [[randint(0, self.topic_num - 1) for n in range(len(self.docs[m]))] for m in range(len(self.docs))]

        self.n_mk = self._init_zero_lls(self.topic_num, len(self.docs))
        self.n_kt = self._init_zero_lls(len(self.words), self.topic_num)
        for m in range(len(self.docs)):
            for k in range(self.topic_num):
                self.n_mk[m][k] = self.state[m].count(k)

        for m in range(len(self.docs)):
            for n in range(len(self.docs[m])):
                word_index = self.words.index(self.docs[m][n])
                topic = self.state[m][n]
                self.n_kt[topic][word_index] += 1


    def _init_zero_lls(self, inner, outer):
        return [[0 for i in range(inner)] for j in range(outer)]


    def _gibbs(self):
        for m in range(len(self.docs)):
            for n in range(len(self.docs[m])):
                self._gibbs_one(m, n)


    def _gibbs_one(self, m, n):
        prob_ranges = [0.0 for k in range(self.topic_num)]
        term_index = self.words.index(self.docs[m][n])
        current_topic = self.state[m][n]

        self.n_mk[m][current_topic] -= 1
        self.n_kt[current_topic][term_index] -= 1

        for k in range(self.topic_num):
            prob_range_deno1 = sum(self.n_kt[k]) + self.beta * len(self.words)
            prob_range_deno2 = sum(self.n_mk[m]) + self.alpha * len(self.docs)

            prob_range_nume1 = self.n_kt[k][term_index] + self.beta
            prob_range_nume2 = self.n_mk[m][k] + self.alpha

            prob_ranges[k] = (prob_range_nume1 * prob_range_nume2) / (prob_range_deno1 * prob_range_deno2)

        new_topic = self._sample_from_range(prob_ranges)
        self.state[m][n] = new_topic
        self.n_mk[m][new_topic] += 1
        self.n_kt[new_topic][term_index] += 1


    def _sample_from_range(self, prob_ranges):
        normal = sum(prob_ranges)
        for k in range(self.topic_num):
            prob_ranges[k] /= normal

        rand = random()
        range_mass = prob_ranges[0]
        for k in range(self.topic_num):
            if rand < range_mass:
                break
            range_mass += prob_ranges[k + 1]

        return k



class lda_res(object):
    def __init__(self, doc_num, word_num, topic_num, phi, theta, words):
        self.doc_num = doc_num
        self.word_num = word_num
        self.topic_num = topic_num
        self.phi = phi
        self.theta = theta
        self.words = words

        self.top_n_words_ls = None
        self.top_n_topic_ls = None


    def top_n_words(self, top_n = None):
        if self.top_n_words_ls == None:
            top_n_words_ls = []
            for k in range(self.topic_num):
                top_words_in_k = [(self.words[t], self.phi[k][t]) for t in range(self.word_num)]
                top_n_words_ls.append(top_words_in_k)
            self.top_n_words_ls = top_n_words_ls

        if top_n == None:
            top_n = self.word_num
        else:
            top_n = min(top_n, self.word_num)

        return [sorted(self.top_n_words_ls[k], key=itemgetter(1), reverse=True)[0:top_n] for k in range(self.topic_num)]


    def top_n_topics(self, top_n = None):
        if self.top_n_topic_ls == None:
            top_n_topic_ls = []
            for m in range(self.doc_num):
                top_topic_in_m = [(k, self.theta[m][k]) for k in range(self.topic_num)]
                top_n_topic_ls.append(top_topic_in_m)
            self.top_n_topic_ls = top_n_topic_ls

        if top_n == None:
            top_n = self.topic_num
        else:
            top_n = min(top_n, self.topic_num)

        return [sorted(self.top_n_topic_ls[m], key=itemgetter(1), reverse=True)[0:top_n] for m in range(self.doc_num)]

