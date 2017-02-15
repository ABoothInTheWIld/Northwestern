# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 23:43:21 2017

@author: Alexander
"""

sentence = 'Sally Sells Seashells by the Seashore'

print('\nThis is the sentence without Sally')

print sentence[6:]

print('\nThis is the sentence without Seashore')

print sentence[:-8]

 

def backwards(word):

    length = len(word)
    
    return_str = ""
    
    index = 1
    
    while index <= length:
    
        letter = word[index*-1]
        
        return_str += letter
        
        index = index + 1

    return return_str

print ('\nThis is the backwards example')
sentence = 'Sally Sells Seashells by the Seashore'
print backwards(sentence)

 

#converting the string into an array

sentence=['Sally ','Sells ','Seashells ','by ','the ','Seashore ']

 

print('\nThis is the sentence backwards with words spelled forward')

print backwards(sentence)

 

def backwards2(sent):

    length = len(sent)
    
    return_str = ""
    
    index = 0
    
    while index < length:
    
        letter = backwards(sent[index])
        
        return_str += letter
        
        index = index + 1

    return return_str

print('\nThis is the words backwards but sentence stays the same')

print backwards2(sentence)