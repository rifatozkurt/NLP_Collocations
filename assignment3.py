# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:29:32 2023

@author: rozku
"""

#%% Tokenization, POS Tagging, Lemmatization

import nltk
from scipy.stats import binom 
from collections import Counter
from math import log
import numpy as np

# nltk.download('popular')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk import pos_tag
#from nltk.corpus import wordnet
import custom_lemmatizer



file = open(r"C:\Users\rozku\Desktop\Student Release\Fyodor Dostoyevski Processed.txt", "r", encoding="utf-8")
corpus = file.read()
file.close()

tokenized = word_tokenize(corpus)
tagged = pos_tag(tokenized, tagset='universal')

cm = custom_lemmatizer.custom_lemmatizer()
lemmatized = []
for token in tagged:
    lemmatized.append((cm.lemmatize(token), token[1]))

#%% Creating Bigrams -- returns bigrams of window size 1 and 3, in the form of a list of tuples

def create_bigrams(tokens, window_size):
    bigrams = []
    for i in range(0, len(tokens)):
        for j in range(1, window_size + 1):
            if i+j < len(tokens):
                bigrams.append((tokens[i], tokens[i + j]))
    return bigrams

bigram_ws1 = create_bigrams(lemmatized, 1)
bigram_ws3 = create_bigrams(lemmatized, 3)

#%% Calculating the frequency of bigrams in a dictionary

def create_freq_dict(bigramlist):
    freq_dict = {}
    for bigram in bigramlist:
        if bigram in freq_dict:
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    return freq_dict

bigram_freq_dict_ws1 = create_freq_dict(bigram_ws1)
bigram_freq_dict_ws3 = create_freq_dict(bigram_ws3)

#%% Filtering bigrams according to POS tags, stop words, frequency and punctuations

def create_filtered_dict(freq_dict):
    file = open(r"C:\Users\rozku\Desktop\Student Release\stopwords.txt", "r", encoding = "utf-8")
    stopwords = file.read().split("\n")
    file.close()
    
    filtered_dict = {}
    filtered_dict_before_cutoff = {}
    for key in freq_dict:
        if (key[0][0].isalpha()) and (key[1][0].isalpha()) and (key[1][1] == 'NOUN') and ((key[0][1] == 'NOUN') or (key[0][1] == 'ADJ')) and (key[0][0] not in stopwords) and (key[1][0] not in stopwords):
            if (freq_dict[key] >= 10):
                if (key[0][0], key[1][0]) not in filtered_dict:
                    filtered_dict[(key[0][0], key[1][0])] = freq_dict[key]
                elif (key[0][0], key[1][0]) in filtered_dict:
                    filtered_dict[(key[0][0], key[1][0])] += freq_dict[key] 
                if (key[0][0], key[1][0]) not in filtered_dict_before_cutoff:
                    filtered_dict_before_cutoff[(key[0][0], key[1][0])] = freq_dict[key]
                elif (key[0][0], key[1][0]) in filtered_dict_before_cutoff:
                    filtered_dict_before_cutoff[(key[0][0], key[1][0])] += freq_dict[key]
            else:
                if (key[0][0], key[1][0]) not in filtered_dict_before_cutoff:
                    filtered_dict_before_cutoff[(key[0][0], key[1][0])] = freq_dict[key]
                elif (key[0][0], key[1][0]) in filtered_dict_before_cutoff:
                    filtered_dict_before_cutoff[(key[0][0], key[1][0])] += freq_dict[key]
                
    return filtered_dict, filtered_dict_before_cutoff
            
bigram_freq_filtered_dict_ws1, bigram_freq_filtered_dict_before_cutoff_ws1 = create_filtered_dict(bigram_freq_dict_ws1)
bigram_freq_filtered_dict_ws3, bigram_freq_filtered_dict_before_cutoff_ws3 = create_filtered_dict(bigram_freq_dict_ws3)


#%% Student's t-test results

def t_tester(tokens, bigram_freq, filename):
    token_freq_dict = Counter(tokens)
    n = len(tokens)

    t_test = {}
    for bigram, freq in bigram_freq.items():

        null = token_freq_dict[bigram[0]] * token_freq_dict[bigram[1]] / n**2
        mle = freq / n
        sample_mean = mle
        t_test[bigram] = (mle - null) / (sample_mean /n)**0.5

    t_test = sorted(t_test.items(), key=lambda x: x[1], reverse=True)
    #write the first 20 bigrams to a file
    with open(filename, "w", encoding='utf-8') as file:
        for bigram, score in t_test[:20]:
            file.write(f"Bigram: {bigram}, Score: {score}, Bigram Count: {bigram_freq[bigram]}, Word Counts: {token_freq_dict[bigram[0]], token_freq_dict[bigram[1]]}\n")
    return t_test

t_test_ws1 = t_tester(tokenized, bigram_freq_filtered_dict_ws1, "t_test_ws1.txt")
t_test_ws3 = t_tester(tokenized, bigram_freq_filtered_dict_ws3, "t_test_ws3.txt")  
    
#%% Pearson's Chi-Square Test results

def chi_square_tester(tokens, bigrams, bigrams_before_cutoff, filename):
    token_freq_dict = Counter(tokens)
    chi_square_test = {}
    
    for key in bigrams:
        n = sum(bigrams_before_cutoff.values())
        o11 = bigrams[key]
        o22 = n - o11
        o21 = 0
        o12 = 0
        
        for key2 in bigrams_before_cutoff:
            if (key[0] == key2[0]) and (key[1] != key2[1]):
                o12 += bigrams_before_cutoff[key2]
            elif (key[0] != key2[0]) and (key[1] == key2[1]):
                o21 += bigrams_before_cutoff[key2]
        chi2 = n * ((o11*o22 - o12*o21) ** 2) / ((o11 + o12)*(o11 + o21)*(o12 + o22)*(o21 + o22))
        
        chi_square_test[key] = (chi2, o11,o12, o21, o22)
    
    chi_square_test = sorted(chi_square_test.items(), key=lambda x: x[1], reverse=True)
    
    with open(filename, "w", encoding='utf-8') as file:
        for bigram, score in chi_square_test[:20]:
            file.write(f"Bigram: {bigram}, Score: {score}, Bigram Count: {bigrams[bigram]}, Word Counts: {token_freq_dict[bigram[0]], token_freq_dict[bigram[1]]}\n")
    return chi_square_test

chi_square_test_ws1 = chi_square_tester(tokenized, bigram_freq_filtered_dict_ws1, bigram_freq_filtered_dict_before_cutoff_ws1, "chi2_test_ws1.txt")
chi_square_test_ws3 = chi_square_tester(tokenized, bigram_freq_filtered_dict_ws3, bigram_freq_filtered_dict_before_cutoff_ws3, "chi2_test_ws3.txt")

#%% Likelihood Ratio Test results

def likelihood_ratio_tester(tokens, bigrams, bigrams_before_cutoff, filename):
    likelihood_test = {} 
    token_freq_dict = Counter(tokens)
    n = sum(bigrams_before_cutoff.values())
    
    for key in bigrams:
        c12 = bigrams[key]
        c1 = token_freq_dict[key[0]]
        c2 = token_freq_dict[key[1]]
        p0 = c2 / n
        ph21 = c12 / c1
        ph22 = (c2- c12) / (n-c1)
        
        likelihood_1 = binom.pmf(c12, c1, p0) * binom.pmf(c2-c12, n-c1, p0)
        likelihood_2 = binom.pmf(c12, c1, ph21) * binom.pmf(c2-c12, n-c1, ph22)
        
        if likelihood_1 == 0:
            likelihood_ratio = 10**-10
        elif likelihood_2 == 0:
            likelihood_ratio = 10**10
        else:
            likelihood_ratio = likelihood_1 / likelihood_2
        
        likelihood_ratio_result = -2 * np.log(likelihood_ratio)
        likelihood_test[key] = (likelihood_ratio_result, c12, c1, c2)
        
        likelihood_test_sorted = sorted(likelihood_test.items(), key=lambda x: x[1], reverse=True)
   
        with open(filename, "w", encoding='utf-8') as file:
              for bigram, score in likelihood_test_sorted[:20]:
                file.write(f"Bigram: {bigram}, Score: {score}, Bigram Count: {bigrams[bigram]}, Word Counts: {token_freq_dict[bigram[0]], token_freq_dict[bigram[1]]}\n")
    return likelihood_test_sorted
    
likelihood_test_ws1 = likelihood_ratio_tester(tokenized, bigram_freq_filtered_dict_ws1, bigram_freq_filtered_dict_before_cutoff_ws1, "likelihood_ratio_tester_ws1.txt")
likelihood_test_ws3 = likelihood_ratio_tester(tokenized, bigram_freq_filtered_dict_ws3, bigram_freq_filtered_dict_before_cutoff_ws3, "likelihood_ratio_tester_ws3.txt")
   
    
    
    
    
    
    
    
    
    
    


