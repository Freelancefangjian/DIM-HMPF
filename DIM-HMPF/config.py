class FLAGES(object):

    pan_size= 16
    ms_size = 4
    hs_size = 1
    
    
    num_spectrum=4
    
    ratio=1
    stride=1
    norm=True
    
    
    batch_size=32
    lr=0.0001
    decay_rate=0.99
    decay_step=10000

    img_path = 'D:\\fangjian\\HSI-MSI-PAN\\chikusei\\dataprocess\\chikusei_train'
    data_path = 'D:\\fangjian\\HSI-MSI-PAN\\chikusei\\dataprocess\\chikusei_train\\train\\train_qk.h5'
    
    is_pretrained=False
    
    iters=500000
    model_save_iters = 500
    valid_iters=10
