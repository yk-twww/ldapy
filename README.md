# LDA with Python
PythonによるLatent Dirichlet Allocation(LDA)の実装。パラメータ推定にはGibbsサンプリングを使用。


## 使い方
```
from ldapy import ldapy

docs = [
    ["aaa", "aaa", "aaa", "aaa", "bbb", "ccc"], #document1
    ["bbb", "bbb", "bbb", "bbb", "ccc", "aaa"], #document2
    ["ccc", "ccc", "ccc", "ccc", "aaa", "bbb"], #document3
    ["aaa", "bbb", "ccc", "aaa", "bbb", "ccc"]  #document4  
]

lda = ldapy()
lda.set(docs, topics = 3, alpha = 3.0, beta = 3.0)
res = lda.estimate(5000)  # 5000 is number of iteration of Gibbs sampling

p_zd = res.top_n_topics() # probability of a document belonging to a topic: P(z|d)
p_wz = res.top_n_words()  # probability of a word being emited from a topic:  P(w|z)
```

```
# for example, p_zd and p_wz are like this
p_zd = [
    [ # about document1
        (1, 0.4),                # to topic1
        (2, 0.3333333333333333), # to topic2
        (0, 0.26666666666666666) # to topic0
    ],
    [ # about document2
        (2, 0.4),
        (0, 0.3333333333333333),
        (1, 0.26666666666666666)
    ],
    [ # about document3
        (1, 0.4666666666666667),
        (0, 0.3333333333333333),
        (2, 0.2)
    ],
    [ # about document4
        (2, 0.4),
        (0, 0.3333333333333333),
        (1, 0.26666666666666666)
    ]
]
p_wz = [
    [ # about topic0
        ('bbb', 0.4375),
        ('aaa', 0.3125),
        ('ccc', 0.25)
    ],
    [ # about topic1
        ('ccc', 0.4444444444444444),
        ('aaa', 0.3888888888888889),
        ('bbb', 0.16666666666666666)
    ],
    [ # about topic2
        ('bbb', 0.4117647058823529),
        ('aaa', 0.29411764705882354),
        ('ccc', 0.29411764705882354)
    ]
]
```