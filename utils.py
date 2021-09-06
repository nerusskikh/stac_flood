import random
import os
import torch
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True
    
    
def colorcode_STAC_errors(
    input: dict,
    output: dict,
    image_key="features",
    target_key="targets",
    output_key="scores",
    max_images=None,
    logits_threshold=0,

                    ):
    """
    Renders STAC predictions with tp as green, fp as red,
    fn as blue and tn as black. Considers masks only.
    """
    
    preds = output[output_key].numpy()
    target = input[target_key].numpy()
    n,_,h,w = preds.shape
    if max_images is not None:
        preds = preds[:max_images,...]
        target = target[:max_images,...]
        n = min(n, max_images)
    preds = preds[:,0,:,:]
    target = target[:,0,:,:]
    res = np.zeros((n,3,h,w), np.uint8)
    res[:,0,:,:] = 255*np.logical_and((preds>=logits_threshold), target==0) #fp
    res[:,1,:,:] = 255*np.logical_and((preds>=logits_threshold), target==1) #tp
    res[:,2,:,:] = 255*np.logical_and((preds<logits_threshold), target==1) #fn
    images = [x for x in res.transpose((0,2,3,1))]
    return images