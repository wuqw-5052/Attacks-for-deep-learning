import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, epoch, model_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,  optimizer, epoch, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(val_loss, model, optimizer, epoch, model_path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,  optimizer, epoch, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, model_path):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        self.val_loss_min = val_loss
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self_state = self.save_self_state()
        model_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch, 'early_stopping_state': self_state}
        torch.save(model_state, model_path)     # 这里会存储迄今最优模型的参数
        # torch.save(model, model_path)                 # 这里会存储迄今最优的模型


    def save_self_state(self):
        self_state = {
            'patience': self.patience,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min,
            'delta': self.delta
        }
        return self_state

    def load_self_state(self, self_state):
        self.patience = self_state['patience']
        self.verbose = self_state['verbose']
        self.counter = self_state['counter']
        self.best_score = self_state['best_score']
        self.early_stop = self_state['early_stop']
        self.val_loss_min = self_state['val_loss_min']
        self.delta = self_state['delta']
