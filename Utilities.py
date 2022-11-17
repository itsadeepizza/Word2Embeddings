# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:30:33 2022

@author: fusco_p
"""

#%%
import requests
import json
import re
import random
import string


from tqdm        import tqdm
from bs4         import BeautifulSoup
from collections import defaultdict


#%%
class getData():
    
    
    #******************************************************************************************************************************************************
    #******************************************************************************************************************************************************
    def __init__( self,
                  max_vocabulary_size ):
        
        self.max_vocabulary_size = max_vocabulary_size
        self.vocabulary          = defaultdict(list)
    
        self.article_generator   = self.gen_article()
        
        
    #******************************************************************************************************************************************************
    #******************************************************************************************************************************************************
    def get_article( self,
                     i,
                     verbose = False ):

        """
        if verbose: print("[wikipedia] putting into file - id "+str(i))
        with open("wikipedia/"+str(i)+"--id.json","w", encoding="utf-8") as f:
            f.writelines(res)
            print("[wikipedia] archived - id "+str(i))
        """
        
        i = int(i)
        if verbose:
            
            print("[wikipedia] getting source - id "+str(i))
            
        link    = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids=" + str(i) + "&inprop=url&format=json"
        text    = requests.get(link).text
        jsonobj = json.loads(text)
        title   = jsonobj['query']['pages'][str(i)]['title']
        if verbose:
            
            print("[wikipedia] link: ",link, " title: ", title)
            
        html   = jsonobj['query']['pages'][str(i)]['extract']
        parsed = BeautifulSoup(html, "html.parser")
        par    = parsed.find_all("p")
        res    = ""
        for element in par:
            
            if element.text.isascii():
                
                res += element.text
                
        return res
    
    #******************************************************************************************************************************************************
    #******************************************************************************************************************************************************
    def gen_article( self ):
        
        for i in range(10000, 1_000_000):
            # i<1000 = api error, <10000 more weird data
            
            try:
                
                art = self.get_article(i, verbose = False)
                if len(art.replace("\n","").replace(" ","")) == 0:  # dont return only whitespaces + \n
                
                    continue
                
                yield art
                
            except KeyError:
                
                continue
            
            
    #******************************************************************************************************************************************************
    #******************************************************************************************************************************************************
    def process_word( self, 
                      word ):
                
        processed_word = re.sub( r'http\S+', ' ', word ).lower()
        processed_word = processed_word.translate( str.maketrans( { key : ' ' for key in string.punctuation } ) )
        processed_word = processed_word.translate( str.maketrans( { key : ' ' for key in string.digits } ) )
        
        return processed_word
        
    
    #******************************************************************************************************************************************************
    #******************************************************************************************************************************************************
    # def getVocabulary( self,
    #                    window_size,
    #                    num_words,
    #                    vocabulary_filename ):
        
    #     while len(self.vocabulary) < self.max_vocabulary_size:
            
    #         #******************************
    #         #GETTING ARTICLE FROM WIKIPEDIA
    #         #******************************
    #         words = []
    #         while len( words ) < num_words:
                
    #             article = next( self.article_generator )
                
    #             #*************************************
    #             #PROCESSING ALL WORDS GOT FROM ARTICLE
    #             #*************************************
    #             word = self.process_word(article)
                
    #             words = word.split( )
                
                
    #         for i in range(len( words ) - window_size):
                
    #             window = words[i:i + window_size]
    #             target = window[2]
                
    #             window[0], window[2] = window[2], window[0]
    #             for element in window[1:]:
                    
    #                 self.vocabulary[target].add(element)
                    
    #             if len(self.vocabulary) == self.max_vocabulary_size:
                    
    #                 break
        
    #         print("Vocabulary Size: ", len( self.vocabulary ))
        
    #     #********************************
    #     # SAVE ALL WORDS IN SEPARATE FILE
    #     #********************************
    #     print("Writing vocabulary")
    #     with open(vocabulary_filename, "w") as fid:
            
    #         for key in self.vocabulary.keys():
            
    #             fid.write(key + "\n")
                
    def getVocabulary( self,
                       window_size,
                       num_words,
                       vocabulary_filename ):
        
        while len(self.vocabulary) < self.max_vocabulary_size:
            
            #******************************
            #GETTING ARTICLE FROM WIKIPEDIA
            #******************************
            words = []
            while len( words ) < num_words:
                
                article = next( self.article_generator )
                
                #*************************************
                #PROCESSING ALL WORDS GOT FROM ARTICLE
                #*************************************
                word = self.process_word(article)
                
                words = word.split( )
                
                
            for i in range(0, len(words) - window_size):
                
                window = words[i:i+window_size]
                target = window[2]
                if target not in self.vocabulary.keys():
                    
                    self.vocabulary[target] = []
                    
                for element in window:
                    
                    if element not in self.vocabulary[target] and element != target:
                        
                        self.vocabulary[target].append(element)
                        if element not in self.vocabulary.keys():
                            
                            self.vocabulary[element] = []
                    
                if len(self.vocabulary) == self.max_vocabulary_size:
                    
                    break
        
            print("Vocabulary Size: ", len( self.vocabulary ))
        
        #********************************
        # SAVE ALL WORDS IN SEPARATE FILE
        #********************************
        print("Writing Vocabulary")
        with open(vocabulary_filename, "w") as fid:
            
            for key in self.vocabulary.keys():
            
                fid.write(key + "\n")
                    

    #******************************************************************************************************************************************************
    #******************************************************************************************************************************************************
    def getDatasetWords( self,
                         dataset_words_filename,
                         num_negative_samples ): # very slow
    
    
        print("Writing Dataset")
        with open(dataset_words_filename, "w") as fid:
            
            fid.write("target" + "; "+ "word" + "; " + "is_close"+"\n")
            for element in tqdm(self.vocabulary, ascii = True, desc = "Dataset"):
                
                for close_word in self.vocabulary[element]:
                    
                    fid.write(element + "; " + close_word + "; " + "1" + "\n")

                    words_basket     = set( self.vocabulary.keys() ).difference( set( self.vocabulary[element] ) )
                    negative_samples = random.sample( words_basket, k = num_negative_samples )
                    
                    #******************************************
                    #WRITE NEGATIVE SAMPLES OUT TO DATASET FILE
                    #******************************************
                    for word in negative_samples: 
                        
                        fid.write(element + "; " + word + "; " + "0" + "\n")
                        

    #******************************************************************************************************************************************************
    #******************************************************************************************************************************************************
    def create_csv( self,
                    window_size,
                    num_words,
                    vocabulary_filename,
                    negative_samples,
                    dataset_words_filename ):
        
        self.getVocabulary( window_size,
                            num_words,
                            vocabulary_filename )
        
        self.getDatasetWords( dataset_words_filename,
                              negative_samples )




#%%
if __name__=="__main__":
    
    VOCAB_SIZE  = 15000
    NEG_SAMPL   = 5     # CONTINOUS BAG OF WORDS WITH NEGATIVE SAMPLES


    data = getData( max_vocabulary_size = VOCAB_SIZE )

    article = data.create_csv( window_size            = 5,
                               num_words              = 10,
                               vocabulary_filename    = "Vocabulary.csv",
                               negative_samples       = NEG_SAMPL,
                               dataset_words_filename = "Dataset.csv" )
        
#%%