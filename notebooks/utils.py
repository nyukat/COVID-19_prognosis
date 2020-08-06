import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import cv2
from PIL import Image
import imageio

class RepeatChannels(object):

    def __init__(self, channels):
        self.channels = channels
        
    def __call__(self, tensor):
        """
        Args:
            tensor: Tensor to be repeated.

        Returns:
            Tensor: repeated tensor.
        """
        return tensor.repeat(self.channels, 1, 1)


    def __repr__(self):
        return self.__class__.__name__ + '()'

preprocess_pipeline = transforms.Compose([
        transforms.CenterCrop((3072, 3072)),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        RepeatChannels(3),
        transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ])

def load_image(image_name):
    # PIL cannot handle 3-channel uint16 images correctly, so using imageio to handle that part
    image = imageio.imread(image_name)
    if image.max() != 0:
        image = image / image.max() # normalize to 0-1 (1-channel image, PIL should still support it)
    image = Image.fromarray(image)
    # preprocess the image
    image = preprocess_pipeline(image)
    return image

# visualize the saliency maps
def visualize_saliency_maps(img, salency_maps):
    fig = plt.figure(figsize=(20,10))
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas
    
    ax = fig.add_subplot(1, 5, 1)
    ax.imshow(img, cmap="Greys_r")
    ax.set_title("original image")
    ax.axis("off")
    
    ax = fig.add_subplot(1, 5, 2)
    ax.imshow(img, cmap="Greys_r")
    show_sm = cv2.resize(salency_maps[0, :, :], (1024, 1024))
    ax.imshow(show_sm, cmap=alpha_red, clim=[0.0, 1.0])
    ax.set_title("saliency map (24 hr)")
    ax.axis("off")
    
    ax = fig.add_subplot(1, 5, 3)
    ax.imshow(img, cmap="Greys_r")
    show_sm = cv2.resize(salency_maps[1, :, :], (1024, 1024))
    ax.imshow(show_sm, cmap=alpha_red, clim=[0.0, 1.0])
    ax.set_title("saliency map (48 hr)")
    ax.axis("off")
    
    ax = fig.add_subplot(1, 5, 4)
    ax.imshow(img, cmap="Greys_r")
    show_sm = cv2.resize(salency_maps[2, :, :], (1024, 1024))
    ax.imshow(show_sm, cmap=alpha_red, clim=[0.0, 1.0])
    ax.set_title("saliency map (72 hr)")
    ax.axis("off")
    
    ax = fig.add_subplot(1, 5, 5)
    ax.imshow(img, cmap="Greys_r")
    show_sm = cv2.resize(salency_maps[3, :, :], (1024, 1024))
    ax.imshow(show_sm, cmap=alpha_red, clim=[0.0, 1.0])
    ax.set_title("saliency map (96 hr)")
    ax.axis("off")
    plt.show()
    

def merge_xray_cbc(xray_pred_vit, cbc_pred, N, hrs=120, ehr_pred=''):
    
    if ehr_pred == '':
        ehr_pred='prob_Adv'+N
    
    if type(N) == list:
        cols = []
        for n in N:
            cols.append('ehr_prob_Adv'+str(n))
        print(cols)
    else:
        cols = ['ehr_prob_Adv'+N]
        
    xray_pred_vit_cbc = xray_pred_vit.merge(cbc_pred[['Patient']+cols], how='left', 
                                    on='Patient')

  
    return xray_pred_vit_cbc

def summarise_results(df_final, N, descrip=''):
    print('Results using Chest X-ray only:')
    test_set_final = df_final.loc[df_final['label_'+N].notna()]
    print('AUROC = {}'.format(roc_auc_score(test_set_final['label_'+N], test_set_final['pred_label_'+N])))
    print('AUPRC = {}'.format(average_precision_score(test_set_final['label_'+N], test_set_final['pred_label_'+N])))

    print('\nResults using clinical variables only:')
    test_set_final = df_final.loc[df_final['label_'+N].notna()]
    print('AUROC = {}'.format(roc_auc_score(test_set_final['label_'+N], test_set_final['ehr_prob_Adv'+N])))
    print('AUPRC = {}'.format(average_precision_score(test_set_final['label_'+N], test_set_final['ehr_prob_Adv'+N])))


    print('\nResults using both modalities:')
    test_set_final = df_final.loc[df_final['label_'+N].notna()]
    print('AUROC = {}'.format(roc_auc_score(test_set_final['label_'+N], test_set_final['final_pred_'+N])))
    print('AUPRC = {}\n'.format(average_precision_score(test_set_final['label_'+N], test_set_final['final_pred_'+N])))
