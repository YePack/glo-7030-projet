import matplotlib.pyplot as plt
from IPython.display import clear_output


class History:

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

    def save(self, train_loss, val_loss, lr):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['lr'].append(lr)

    def add_history(self, history_add):
        self.history['train_loss'] += history_add.history['train_loss']
        self.history['val_loss'] += history_add.history['val_loss']
        self.history['lr'] += history_add.history['lr']

    def display_loss(self):
        epoch = len(self.history['train_loss'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(epochs, self.history['train_loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Validation')
        plt.legend()
        plt.show()

    def display_lr(self):
        epoch = len(self.history['train_loss'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Learning rate')
        plt.xlabel('Epochs')
        plt.ylabel('Lr')
        plt.plot(epochs, self.history['lr'], label='Lr')
        plt.show()

    def display(self):
        epoch = len(self.history['train_loss'])
        epochs = [x for x in range(1, epoch + 1)]

        fig, axes = plt.subplots(2, 1)
        plt.tight_layout()

        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['train_loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')

        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Lr')
        axes[2].plot(epochs, self.history['lr'], label='Lr')

        plt.show()
