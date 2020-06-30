import os
import numpy as np
import statistics as st

# distintic keys from files

# ACNN = 'ACNN'
unet = 'Unet'
# encoder = 'encoder'
doble_unet = 'doble_unet'

# values that we want to get the mean
switcher = {
    # ACNN: 'val_decode_Activation_unet_dice',
    unet: 'val_dice',
    # encoder: 'val_dice',
    doble_unet: 'val_UNET_dice',
}

# From which point would you like to start calculating the mean
FROM = 150

# directory where results are
# rootdir = os.getcwd()
rootdir = '/home/andres/Documents/TESIS/tesis-andres-cell/RESULTS/results-Unet'

tmp = dict()
lista = list()
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if 'history' in file:
            tmp = np.load(os.path.join(subdir,file), allow_pickle=True).item()
            #if ACNN in file:
            #    lista.append(subdir + ": " + str(st.mean(tmp[switcher.get(ACNN, "Invalid key")][FROM:])) )
            if unet in file:
                lista.append(subdir + ": " + str(st.mean(tmp[switcher.get(unet, "Invalid key")][FROM:])) )
            #elif encoder in file:
            #    lista.append(subdir + ": " + str(st.mean(tmp[switcher.get(encoder, "Invalid key")][FROM:])) )
            elif doble_unet in file:
                print(subdir + ": " + str(st.mean(tmp[switcher.get(doble_unet, "Invalid key")][FROM:])) )
lista.sort()
for i in lista:
    print(i)

"""
doble_unet_4: 0.84958386
doble_unet_5: 0.8532624
doble_unet_1: 0.8471107
doble_unet_3: 0.8506013
doble_unet_2: 0.8515701


unet_unet_1: 0.8489727
unet_unet_2: 0.85079384
unet_unet_3: 0.8485278
unet_unet_4: 0.8525475
unet_unet_5: 0.85346663


results_doble_unet_5: 0.8510769106092907
results_doble_unet_5: 0.857960847022639
results_doble_unet_3: 0.8559799921873844
results_doble_unet_3: 0.8596864910600167
results_doble_unet_4: 0.8551680135847343
results_doble_unet_4: 0.8580293004357036
results_doble_unet_2: 0.8583626907837542
results_doble_unet_2: 0.8530147660246082
results_doble_unet_1: 0.8600139379511453
results_doble_unet_1: 0.8517220456226199
results_ACNN_1: 0.42181677
results_ACNN_2: 0.44481006
results_ACNN_3: 0.4434296
results_ACNN_4: 0.45677784
results_ACNN_5: 0.4512096
results_encoder_1: 0.9915793
results_encoder_2: 0.99172205
results_encoder_3: 0.99173737
results_encoder_4: 0.9917837
results_encoder_5: 0.991747
results_unet_1: 0.8463594
results_unet_2: 0.85083777
results_unet_3: 0.8480702
results_unet_4: 0.85185957
results_unet_5: 0.8528409
"""