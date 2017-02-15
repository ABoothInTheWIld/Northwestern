# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:28:15 2017

@author: Alexander
"""

#String Manipulation Practice
#MSPA Week 7 Discussion 2

#Master String: ‘Sally Sells Seashells by the Seashore’
masterString = "Sally Sells Seashells by the Seashore"

#1. No Sally: ‘Sells Seashells by the Seashore’
words = masterString.split(' ')  # Split on whitespace
words.remove("Sally")
string_no_sally = " ".join(words)
print(string_no_sally)
#Check:
#Sells Seashells by the Seashore

#2. No Seashore: ‘Sally Sells Seashells’
words = masterString.split(' ')  # Split on whitespace
words.remove("by")
words.remove("the")
words.remove("Seashore")
string_no_seashore = " ".join(words)
print(string_no_seashore)
#Check:
#Sally Sells Seashells

#3. Write the string backwards: ‘erohsaeS eht yb sllehsaeS slleS yllaS’
reversedMaster = ''.join(reversed(masterString))
print(reversedMaster)
#Check:
#erohsaeS eht yb sllehsaeS slleS yllaS

#4. Write the sentence backwards, keeping the words spelled forward:
#‘Seashore the by Seashells Sells Sally’
words = masterString.split(' ')  # Split on whitespace
wordsReversed = [x for x in list(reversed(words))]
string_reversed_words = " ".join(wordsReversed)
print(string_reversed_words)
#Check:
#Seashore the by Seashells Sells Sally

#5. Write the words backwards, but keep the sentence the same.
#‘yllaS slleS sllehsaeS yb eht erohsaeS’
words = masterString.split(' ')  # Split on whitespace
wordsReversedIndiv = [x[::-1] for x in words]
string_reversed_words_indiv = " ".join(wordsReversedIndiv)
print(string_reversed_words_indiv)
#Check:
#yllaS slleS sllehsaeS yb eht erohsaeS