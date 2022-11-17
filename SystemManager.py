# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:34:03 2022

@author: fusco_p
"""

#%%

import torch
import torch.nn as nn


import shutil

#%%
class Manager():
    
    def __init__( self,
                  model,
                  criterion,
                  optimizer,
                  dataloader,
                  epochs,
                  device,
                  model_file = None,
                  save_freq  = 1000,
                  plot_freq  = 1000 ):
        
        
        self.model        = model
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.dataloader   = dataloader
        self.epochs       = epochs
        self.device       = device
        self.model_file   = model_file
        self.model_folder = model_file.parent
        self.save_freq    = save_freq
        self.plot_freq    = plot_freq
        
        if self.model_folder.is_dir():
            
            shutil.rmtree( self.model_folder )

        self.model_folder.mkdir(parents = True)
        
    
    def train(self):
        
    
        #**************
        # TRAINING LOOP
        #**************
        tot_iters = 0
        for epoch in range(self.epochs):
            
            print("Epoch started...: ", epoch)
            # data = iter(self.dataloader)
            # for i, (words, label) in enumerate(data):
            for i, (words, label) in enumerate(self.dataloader, 0):
                
                out  = self.model(words)
                loss = self.criterion(out, label.to(self.device))
                
                if i % self.plot_freq == 0:
                    
                    print(f"Epoch {epoch} iteration {i} loss:", loss.item())
                    
                    # if use_tensorboard:
                        
                    #     index    = epoch*tot_iters + i
                    #     plot_buf = gen_plot(model.embedding, "is", "was", "airport", index)
                    #     image    = PIL.Image.open(plot_buf)
                    #     image    = ToTensor()(image).unsqueeze(0)[0]
                        
                    #     writer.add_image('plot embedding', image, index)
                    #     writer.add_scalar('train_loss', loss.item(), index)
                        
                if ( i % self.save_freq == 0 ) & ( self.model_file is not None ):
                    
                    torch.save( self.model.state_dict(), self.model_file )
                    print( f"\t Model <{self.model_file.name}> saved to folder : {self.model_folder.name}\n")
                    
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if i > tot_iters:
                    
                    tot_iters = i
                    
            # TODO save model / PCA prima di plottare
    
    
    def test(self):
        
        pass
#%%