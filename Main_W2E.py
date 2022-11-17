# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:29:10 2022

@author: fusco_p
"""

#%%

import torch
import torch.nn as nn
import pathlib
#%%
from torch.utils.data import DataLoader


from Utilities     import getData
from Model         import CustomModel
from Dataset       import CustomDataset
from SystemManager import Manager


#%%

if __name__=="__main__":
    
#%%

    VOCAB_SIZE             = 5000
    EMBEDDING_SIZE         = 50
    NEG_SAMPL              = 5     # CONTINOUS BAG OF WORDS WITH NEGATIVE SAMPLES
    VOCABULARY_FILENAME    = "Vocabulary.csv"
    DATASET_WORDS_FILENAME = "Dataset.csv"
    BATCH_SIZE             = 1
    DEVICE                 = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    EPOCHS                 = 2
    LEARNING_RATE          = 1e-4
    SAVE_MODEL_FOLDER      = pathlib.Path.cwd() / 'MODEL_SAVED'
    SAVE_MODEL_FILENAME    = SAVE_MODEL_FOLDER / 'Model.pth'
    CREATE_DATASET_WORDS   = True
    
#%%

    if CREATE_DATASET_WORDS:
        
        data = getData( max_vocabulary_size = VOCAB_SIZE )
        
    
        data.create_csv( window_size            = 5,
                         num_words              = 10,
                         vocabulary_filename    = VOCABULARY_FILENAME,
                         negative_samples       = NEG_SAMPL,
                         dataset_words_filename = DATASET_WORDS_FILENAME )
        
    
#%%
    
    dataset_words    = CustomDataset( DATASET_WORDS_FILENAME, 
                                      VOCABULARY_FILENAME )
    
    dataloader_words = DataLoader( dataset    = dataset_words, 
                                    batch_size = BATCH_SIZE, 
                                    shuffle    = True)
    
#%%
    
    model = CustomModel( device         = DEVICE, 
                         len_vocabulary = VOCAB_SIZE, 
                         embedding_size = EMBEDDING_SIZE)
    
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)

#%%
    
    manager = Manager( model,
                        criterion,
                        optimizer,
                        dataloader_words,
                        EPOCHS,
                        DEVICE,
                        SAVE_MODEL_FILENAME )
    
#%%

    manager.train()
    
#%%
