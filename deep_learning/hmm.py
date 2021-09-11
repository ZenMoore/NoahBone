import nltk
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, ConditionalProbDist

"""
There are three basic problems : (1)probability calculation, (2)parameter learning and (3)prediction.

supervised learning -- labeled dataset -- get parameters by statistics.
unsupervised learning -- unlabeled dataset -- get parameters by EM-algorithm or by back propagation.

RIGHT NOW, there is only supervised version of HMM in this file.
Learning Material 1 (2): https://github.com/qingyujean/Magic-NLPer/blob/main/MachineLearning/HMM%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B/t1.ipynb
Learning Material 2 (1)&(3): https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC10%E7%AB%A0%20%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B/10.HMM.ipynb

For unsupervised case with EM:
with probability calculation, we can get it easily (just see the math derivation outcome of the algo !)

For unsupervised case with back propagation:
a little bit puzzled, please see https://github.com/lorenlugosch/pytorch_HMM/blob/master/README.md and learn it !
in fact, this is only min-neg-log-likelihood(loss) estimation, hence also easy.
"""

# todo : unsupervised case

'special tokens'
start_token = '<start>'
end_token = '<end>'
# why need we these two special tokens ?
# see direct calculation of probability, P(w0) is in fact P(w0|start_token), with P(start_token) = 1
# hence, this facilitates the calculation of P(w0).

'dataset information'
# what type ?
print(type(brown.tagged_sents()[0]), end=' : type(brown.tagged_sents()[0])\n')
# how does it like ?
print(brown.tagged_sents()[0], end=' : brown.tagged_sents()[0]\n')  # [(word, tag), (word, tag), (word, tag)...]

'dataset arrangement'
dataset = []  # we need to arrange all the sentences in brown corpus together.
for sent in brown.tagged_sents():
    dataset.append((start_token, start_token))
    # permute word and tag, so as to facilitate the following use of nltk.ConditionalFreqDist
    dataset.extend([(tag, word) for (word, tag) in sent])
    dataset.append((end_token, end_token))
print(len(dataset), end=' : len(dataset)\n')

'param learning : statistics : observation probability distribution'
# for the calculation of conditional probability distribution (P(w|t))
# nltk has ConditionalFreqDist
# 1. by object : cfdist[condition][word] (condition is t, word is w)
# 2. by tuple list : ConditionalFreqDist([(tag, word), (tag, word), (tag, word)...])
# by this class, we can easily calculate conditional probability distribution
opd_dist = ConditionalFreqDist(dataset)  # get statistical frequency
# {'condition1':FreqDist({'word': freq, ...}), 'condition2':FreqDist(...)}
opd_dist = ConditionalProbDist(opd_dist, nltk.MLEProbDist)  # get probability (if MLE, equivalent to relative frequency)
print(opd_dist['AT'].prob('The'), end=' : P("The"|"AT")\n')  # P('The'|'AT')

'param learning : statistics : transition probability distribution'
brown_tags = [tag for (tag, word) in dataset]
tpd_dist = ConditionalFreqDist(nltk.bigrams(brown_tags))
tpd_dist = ConditionalProbDist(tpd_dist, nltk.MLEProbDist)
print(tpd_dist['AT'].prob('NP-TL'), end=" : P('NP-TL'|'AT')\n")


'test examples for probability calculation'
words = 'I want to race'
words = [start_token] + words.split(' ') + [end_token]
tags = 'PP VB TO VB'
tags = [start_token] + tags.split(' ') + [end_token]
all_tags = list(opd_dist.keys())
num_type_tags = len(all_tags)


'probability calculation : direct calculation'
# because P(w) is impossible to be calculated by this method (\sum_t{P(w, t)} is very complex)
# we just calculate P(w, t)
def probability_direct(words, tags):
    res = 1
    for i in range(len(words) - 1):
        a = tpd_dist[tags[i]].prob(tags[i+1])
        if tags[i+1] == end_token:
            b = 1
        else:
            b = opd_dist[tags[i+1]].prob(words[i+1])
        res *= a * b
        # print('P(%s|%s) * P(%s|%s) = %f * %f = %f' % (tags[i+1], tags[i], words[i+1], tags[i+1], a, b, a * b))
    return res


print(probability_direct(words, tags), end=' : probability_direct\n')


'probability calculation : forward'
def probability_forward(words):
    T = len(words)
    N = num_type_tags
    alphas = [[0] *  N] * T
    # t = 0, we get initial states
    for i in range(N):
        a0i = tpd_dist[start_token].prob(all_tags[i])
        # bii = opd_dist[all_tags[i]].prob(words[0])
        # print('P(%s|%s) = %f' % (words[0], all_tags[i], bii))
        alphas[0][i] = a0i

    # t = 1, we get initial recursion
    for i in range(N):
        alphas[1][i] = alphas[0][i] * opd_dist[all_tags[i]].prob(words[1])

    for t in range(1, T-2):  # drop end_token
        for i in range(N):
            alphas[t+1][i] = sum([alphas[t][j] * tpd_dist[all_tags[j]].prob(all_tags[i]) for j in range(N)]) * opd_dist[all_tags[i]].prob(words[t+1])
    return sum(alphas[T-2])  # drop end_token


print(probability_forward(words), end=' : probability_forward\n')


'probability calculation : backward'
def probability_backward(words):
    T = len(words)
    N = num_type_tags
    betas = [[0] * N] * (T-1)  # drop end_token

    pis = []
    # t = 0, we get initial states
    for i in range(N):
        pii = tpd_dist[start_token].prob(all_tags[i])
        pis.append(pii)

    # t = T - 1, initial state (for backward propagation)
    for i in range(N):
        betas[T - 2][i] = 1
    for t in range(T-3, 0, -1):
        total = 0
        for j in range(N):
            aij = tpd_dist[all_tags[i]].prob(all_tags[j])
            bjtp1 = opd_dist[all_tags[j]].prob(words[t+1])
            total += aij * bjtp1 * betas[t+1][j]
        betas[t][i] = total
    total = 0
    for i in range(N):
        total += pis[i] * opd_dist[all_tags[i]].prob(words[1]) * betas[1][i]
    return total

print(probability_backward(words), end=' : probability_backward\n')


'prediction : viterbi' # todo :
def viterbi(words):
    T = len(words)
    N = num_type_tags
    deltas = [[0] * N] * T
    psis = [[0] * N] * T
    pis = []
    for i in range(N):
        pii = tpd_dist[start_token].prob(all_tags[i])
        pis.append(pii)

    # initial recursion
    for i in range(N):
        deltas[1][i] = pis[i] * opd_dist[all_tags[i]].prob(words[1])
        psis[1][i] = 0

    for t in range(2, T - 1):
        for i in range(N):
            temp = [deltas[t - 1][j] * tpd_dist[all_tags[j]].prob(all_tags[i]) for j in range(N)]
            maximum = max(temp)
            deltas[t][i] = maximum * opd_dist[all_tags[i]].prob(words[t])
            psis[t][i] = temp.index(maximum)

    prob = max(deltas[T-2])
    iT = deltas[T - 2].index(prob)

    res = [iT]
    former = iT
    for t in range(T-3, 0, -1):
        res.append(psis[t+1][res[-1]])
        print(all_tags[psis[t+1][res[-1]]])
    print(len(res))  # T - 2, drop begin_token and end_token
    res.reverse()
    return [all_tags[i] for i in res]

print(len(words))
print(' '.join(words[1:-1]) + ' : ' + ' '.join(viterbi(words)), end=' : viterbi\n')
