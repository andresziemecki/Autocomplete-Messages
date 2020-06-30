import os
import numpy as np
import matplotlib.pyplot as plt



def load_history(folder, metric, fname):
    data = []
    for i in range(1,6):
        dfile = os.path.join(folder + '_{}'.format(i), fname)
        data_i = np.load(dfile, allow_pickle=True).item()
        data.append(data_i[metric])
    return np.asarray(data)




def plot_kfold(ax, data, ls ='-', lw=2, color='k', label=None, alpha=.2,
        only_mean=False):
    x = range(data.shape[1])
    ax.plot(x, data.mean(axis=0), ls=ls, color=color, label=label, lw=lw)

    if not only_mean:
        ax.fill_between(x, data.min(axis=0), data.max(axis=0),
                facecolor=color, alpha=.5)
        # Linea negra del borde
        plt.plot(x,data.max(axis=0), 'k-', alpha=alpha)
        plt.plot(x, data.min(axis=0), 'k-', alpha=alpha)




#---- Load data
FHISTORY = './'
metric = {
        'acnn': 'val_decode_Activation_unet_dice',
        'doble': 'val_dice',
        'unet': 'val_dice',
        'autoencoder': 'val_dice',
        }

folder = {
        'acnn': os.path.join('ACNN','results_ACNN'),
        'doble': os.path.join('doble-unet','results'),
        'unet': os.path.join('ACNN','results_unet'),
        'autoencoder': os.path.join('ACNN','results_encoder'),
        }
hname = {
        'acnn':'ACNN_history.npy',
        'doble':'dobleUnet_history.npy',
        'unet':'Unet_history.npy',
        'autoencoder':'autoencoder_history.npy',
        }


acnn = load_history(folder['acnn'], metric['acnn'], hname['acnn'])
unet = load_history(folder['unet'], metric['unet'], hname['unet'])
autoencoder = load_history(folder['autoencoder'], metric['autoencoder'], hname['autoencoder'])
#  doble = load_history(folder['doble'], metric['doble'], hname['doble'])




#---- Plot
f = plt.figure()
plt.title('Accuracy')
ax = f.get_axes()[0]
plot_kfold(ax, acnn, ls='-', color='r', lw=1.5, alpha=.1,
        only_mean=False, label="ACNN")
#  plt.legend()
ax.set_xlabel('Number of epochs')
ax.grid(True)
ax.set_ylabel("Accuracy")
f.tight_layout()
#  f.savefig('p4-5_val_acc.pdf')


f = plt.figure()
plt.title('Accuracy')
ax = f.get_axes()[0]
plot_kfold(ax, autoencoder, ls='-', color='r', lw=1.5, alpha=.1,
        only_mean=False, label="Autoencoder")
#  plt.legend()
ax.set_xlabel('Number of epochs')
ax.grid(True)
ax.set_ylabel("Accuracy")
f.tight_layout()


f = plt.figure()
plt.title('Accuracy')
ax = f.get_axes()[0]
plot_kfold(ax, unet, ls='-', color='r', lw=1.5, alpha=.1,
        only_mean=False, label="Unet")
#  plt.legend()
ax.set_xlabel('Number of epochs')
ax.grid(True)
ax.set_ylabel("Accuracy")
f.tight_layout()
