# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:01:46 2022

@author: fusco_p
"""
#%%
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#%%

class CustomDataset(Dataset):

    def __init__( self, 
                  dataset_words_filename,
                  vocabulary_filename ):
        
        try:
            
            self.dataset_words = pd.read_csv(dataset_words_filename, sep = ";", skipinitialspace = True)
            self.num_samples   = len(self.dataset_words)
            
        except FileNotFoundError:
            
            print(f"cant open file --> {dataset_words_filename}")

        try:
            
            self.vocabulary_words = pd.read_csv(vocabulary_filename, sep = ";", header = None).values.squeeze()
            self.vocabulary_dim   = len( self.vocabulary_words )
            
        except FileNotFoundError:
            
            print(f"cant open file --> {vocabulary_filename}")
            
        
        #**************************************************
        #CREATING THE VOCABULARY INCLUDING ONE HOT ENCODING
        #**************************************************
        self.vocabulary = { key : self.one_hot(idx) for idx, key in enumerate( self.vocabulary_words, 0 ) }
        
            
    def one_hot( self,
                 index ):
        
        result        = torch.zeros(self.vocabulary_dim)
        result[index] = 1
        
        return result
            

    def __getitem__( self, index ):
        
        try:
            
            dataset_words_row = self.dataset_words.iloc[index].to_list()
            embeddings_word   = self.vocabulary[ dataset_words_row[0] ]
            context_word      = self.vocabulary[ dataset_words_row[1] ]
            close_label       = dataset_words_row[2]
            
        except KeyError:
            
            print('KeyError encountered')
            
        result = ( torch.stack( ( embeddings_word, context_word) ), torch.Tensor( [ close_label ] ) )
        
        return result
    
        
    def __len__( self ):
        
        return self.num_samples
    
    
#%%
if __name__=="__main__":
    
    VOCABULARY_FILENAME    = "Vocabulary.csv"
    DATASET_WORDS_FILENAME = "Dataset.csv"
    BATCH_SIZE             = 1
    
    dataset_words    = CustomDataset( DATASET_WORDS_FILENAME, VOCABULARY_FILENAME )
    dataloader_words = DataLoader(dataset = dataset_words, batch_size = BATCH_SIZE, shuffle = True)
    
    out = iter(dataloader_words)
    val = next(out)
    
#%%

