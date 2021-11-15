#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                curLine = 0
                for row in f.readlines():
                    curRow = row.strip().split(' ')
                    if curRow == ["-1"]:
                        break
                    elif curLine < self.unique_words:
                        index, word, count = curRow[0], curRow[1], curRow[2]
                        self.word[index] = word
                        self.index[word] = index
                        self.unigram_count[word] = count
                    else:
                        first, second, probability = curRow[0], curRow[1], curRow[2]
                        self.bigram_prob[self.word[first]][self.word[second]] = probability
                    curLine += 1
            return True

        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def compute_entropy_cumulatively(self, word):
        # YOUR CODE HERE
        l1 = self.lambda1
        l2 = self.lambda2
        l3 = self.lambda3

        curAdd = l3     # initiera curAdd med lambda 3
        if word in self.index:  # check if word is in training model
            wordProbability = int(self.unigram_count[word]) / self.total_words

            if self.last_index == -1:   # if previous word was not in training model
                curAdd += l2 * wordProbability

            else:   # if previous word was in training model
                previousWord = self.word[self.last_index]
                if word in self.bigram_prob[previousWord]:   # if word in previous word bigram
                    bigramProbability = math.exp(float(self.bigram_prob[previousWord][word]))
                    curAdd += l1 * bigramProbability

                else:
                    curAdd += l2 * wordProbability

            self.last_index = self.index[word]

        else:   # if word is not in training model
            self.last_index = -1

        self.logProb += math.log(curAdd)    # lägg på curAdd på logProb
        self.test_words_processed += 1


    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) # Important that it is named self.tokens for the --check flag to work
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)

                self.logProb = -(1 / self.test_words_processed) * self.logProb

            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')
    parser.add_argument('--check', action='store_true', help='check if your alignment is correct')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    if arguments.check:
        results  = bigram_tester.logProb

        payload = json.dumps({
            'model': open(arguments.file, 'r').read(),
            'tokens': bigram_tester.tokens,
            'result': results
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab2_tester',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))
            print('Success! Your results are correct')
        else:
            print('Your results:')
            print('Estimated entropy: {0:.2f}'.format(bigram_tester.logProb))
            print("The server's results:\n Entropy: {0:.2f}".format(response_data['result']))

    else:
        print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()
