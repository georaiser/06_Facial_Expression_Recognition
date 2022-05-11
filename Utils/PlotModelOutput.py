#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:44:46 2022

@author: hellraiser
"""
import matplotlib as plt
class PlotModel():
    def __init__(self, history, name='model'):
        super(PlotModel, self).__init__()
        self.history=history
        self.name=name

    # plot output function
    def plot_model(self):       
        loss_values = self.history['train_loss']
        val_loss_values = self.history['val_loss']
        accuracy_values = self.history['train_acc']
        val_accuracy_values = self.history['val_acc']
              
        fig = plt.figure(figsize=(19,3))
        plt.subplot(1, 2, 1)
        plt.suptitle(self.name, fontsize=18)
        plt.title('loss')
        epoch = range(1,len(loss_values)+1)
        plt.plot(epoch,loss_values, '--',label='loss')
        plt.plot(epoch,val_loss_values, '--',label='val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        
        plt.subplot(1, 2, 2)
        plt.suptitle(self.name, fontsize=18)
        plt.title('accuracy')
        epoch = range(1,len(loss_values)+1)
        plt.plot(epoch,accuracy_values, '--',label='accuracy')
        plt.plot(epoch,val_accuracy_values, '--',label='val_accuracy')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()