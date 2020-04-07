import pandas as pd
import nltk
import numpy as np
import os

START_TOKEN = '<START>'
END_TOKEN = '<END>'

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################### Start of Data Loader functions #######################

def CONLL2003Download():
    # For NER
    data_format = '''
This is CONLL 2003 dataset https://www.clips.uantwerpen.be/conll2003/ner/
Returns: A dictonary of 3 DataFrame (train,val and test) with cols
Word\t POS Tag\t syntactic chunk tag\t named entity tag\t line number
The data contains multiple lines given in the line number column.
'''
    print(data_format)
    os.system('wget -nc "https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/test.txt" -O "test.txt" ')
    os.system('wget -nc "https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/train.txt" -O "train.txt" ')
    os.system('wget -nc "https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003/valid.txt" -O "valid.txt" ')

    with open ('test.txt') as f:
        next(f)
        next(f)
        test = f.read()
        test = test.split('\n\n')
        test = [l.split('\n') for l in test]
        
        test_data = []
        for i,l in enumerate(test):
            for w in l:
                vals = w.split(' ')
                tmp = {}
                tmp['word'] = vals[0]
                tmp['pos'] = vals[1]
                tmp['sct'] = vals[2]
                tmp['net'] = vals[3]
                tmp['line'] = i
                test_data.append(tmp)
        test_df = pd.DataFrame(test_data)

    with open ('train.txt') as f:
        next(f)
        next(f)
        train = f.read()
        train = train.split('\n\n')
        train = [l.split('\n') for l in train]
        
        train_data = []
        for i,l in enumerate(train):
            for w in l:
                vals = w.split(' ')
                tmp = {}
                tmp['word'] = vals[0]
                tmp['pos'] = vals[1]
                tmp['sct'] = vals[2]
                tmp['net'] = vals[3]
                tmp['line'] = i
                train_data.append(tmp)
        train_df = pd.DataFrame(train_data)

    with open ('valid.txt') as f:
        next(f)
        next(f)
        val = f.read()
        val = val.split('\n\n')
        val = [l.split('\n') for l in val]
        
        val_data = []
        for i,l in enumerate(val):
            for w in l:
                vals = w.split(' ')
                tmp = {}
                tmp['word'] = vals[0]
                tmp['pos'] = vals[1]
                tmp['sct'] = vals[2]
                tmp['net'] = vals[3]
                tmp['line'] = i
                val_data.append(tmp)
        val_df = pd.DataFrame(val_data)

    data = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    return data

###################### End of Data Loader functions #######################

# GetData Function
# [Additional datasets](https://github.com/awesomedata/awesome-public-datasets) 

def getData(task = 'languageModeling',specific = None):
    ################################################
    # NLTK corpus 
    print('Additional datasets: https://github.com/awesomedata/awesome-public-datasets')
    # from nltk.corpus import reuters
    # nltk.download('reuters')
    # return reuters.raw(categories='crude' )

    if(task == 'tagging'):
        if(specific == None or specific == 'NER'):
            return CONLL2003Download()

    if(task == 'classification'):
        if(specific == None or specific == 'Sentiment' ):
            # IMDB Movie Review Sentiment Classification (stanford)
            raise Exception("This dataset is not created")


        if(specific == 'Topic'):
            # Reuters Newswire Topic Classification (Reuters-21578)
            raise Exception("This dataset is not created")

    if(task == 'languageModeling'):
        import nltk
        if('specific' == 'gutenberg'):
            from nltk.corpus import gutenberg
            nltk.download('gutenberg')
            file_name ='austen-emma.txt'
            return gutenberg.raw(file_name)
        if('specific' == 'brown'):
            from nltk.corpus import brown


    if(task == 'imageCaptioning'):
        raise Exception("This dataset is not created")
    
    if(task == 'machineTranslation'):
        raise Exception("This dataset is not created")

    if(task == 'questionAnswering'):
        raise Exception("This dataset is not created")

    if(task == 'speechRecognition'):
        raise Exception("This dataset is not created")

    if(task == 'textSummarization'):
        raise Exception("This dataset is not created")

    if(task == 'textSummarization'):
        raise Exception("This dataset is not created")

######################## Text2Words #########################

'''
---
**ordinary text datasets require** \\
1. tokenization 
    * *simple* sentence tokenization 
```
from nltk.tokenize import sent_tokenize 
sent = sent_tokenize(text)
```
    * *other than english* sentence tokenization
```
import nltk.data 
tokenizer = nltk.data.load('tokenizers/punkt/PY3/language.pickle')   
sent = tokenizer.tokenize(text) 
```
    * word tokenization 
```
from nltk.tokenize import word_tokenize   
wordsList = word_tokenize(text) 
``` 
2. stop word removel 
```
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
filtered_sentence = [w for w in stemmed_words if not w in stop_words] 
```
3. stemming **or** lemmatization \\
stemming better than lemetization [reference](https://stackoverflow.com/questions/49354665/should-i-perform-both-lemmatization-and-stemming) 
```
## Stemming
# liking,likely -> like
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
stemmed_words = []
for w in wordsList: 
    stemmed_words.append(ps.stem(w)) 
```
```
## lemmatization
# better -> good
from nltk.stem import WordNetLemmatizer   
lemmatizer = WordNetLemmatizer() 
print("rocks :", lemmatizer.lemmatize("rocks")) 
print("corpora :", lemmatizer.lemmatize("corpora")) 
# a denotes adjective in "pos" for better result 
print("better :", lemmatizer.lemmatize("better", pos ="a")) 
```
---

'''

import re
def isfloat(el):
    if re.match(r'^-?\d+(?:\.\d+)?$', el) is None:
        return False
    return True

def text2words(text,bag_of_words = False):
    # input list of text
    # output list of list of word tokens
    from nltk.tokenize import word_tokenize   
    from nltk.corpus import stopwords 
    from nltk.stem import PorterStemmer 
    import nltk

    nltk.download('punkt')
    nltk.download('stopwords')
    
    word_tolkens = []
    for docs in text:
        # tokeninzation
        wordsList = word_tokenize(docs)

        wordsList = [('99' if isfloat(w) else w.lower()) for w in wordsList]

        if(bag_of_words):
            # stopwords removel 
            stop_words = set(stopwords.words('english')) 
            filtered_sentence = [w.lower() for w in wordsList if not w in stop_words] 

            # stemming
            ps = PorterStemmer() 
            stemmed_words = [ps.stem(w) for w in filtered_sentence]

            word_tolkens.append(stemmed_words)
        else:
            word_tolkens.append(wordsList)

    return word_tolkens

def tokens2vocab(words):
    if(type(words[0]) != str):
        return list(set([w for docs in words for w in docs]))
    else:
        return list(set(words))
    

####################################################################

'''
### Words to Embeddings
---
gensim module [referene](https://github.com/RaRe-Technologies/gensim-data) 

* Word2Vec is generated using CBOW or SkipGram 
* GLoVe is generated using co-occurrence matrix and ML

'''


def load_embedding_model(dataset='glove-wiki-gigaword-50'):

    import gensim.downloader as api

    wv_from_bin = api.load(dataset)
    # word2vec-google-news-300

    print("Loaded vocab size %i" % len(wv_from_bin.vocab.keys()))
    return wv_from_bin

def word2embedding(docs):
    wv_from_bin = load_embedding_model()
    embedding_vocab = list(wv_from_bin.vocab.keys())
    data_vocab = tokens2vocab(docs)
    dim = wv_from_bin.vector_size
    new_words = 0
    for w in data_vocab:
        if(w not in embedding_vocab):
            wv_from_bin.add(w,np.random.rand(dim))
            new_words+=1
    print(str(new_words)+" new words are added")
    return wv_from_bin


'''
---
* To get the vector of a word
```
wv_from_bin.get_vector(word)
```
* To get the most similar words
```
wv_from_bin.most_similar(word)
```
---
'''
