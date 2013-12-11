# -*- coding: utf-8 -*-

from pprint import PrettyPrinter
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src")

from ldapy import ldapy 


if __name__ == "__main__":

    docs = [
        ["ant", "ant", "ant", "ant", "ant", "ant", "ant", "ant", "bat", "dog"],
        ["bat", "bat", "bat", "bat", "bat", "bat", "bat", "bat", "dog", "ant"],
        ["dog", "dog", "dog", "dog", "dog", "dog", "dog", "dog", "ant", "bat"],
    ]

    lda = ldapy()
    lda.set(docs, 3, 3.0, 3.0)
    result = lda.estimate(50000)

    pp = PrettyPrinter(indent=2)
    pp.pprint(result.top_n_words())
    pp.pprint(result.top_n_topics())   
