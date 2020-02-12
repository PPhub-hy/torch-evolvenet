"""
Load CIFAR-10 dataset and generate batch-wise data stream for training image classification task.
"""

import os
import numpy as np
import pickle
#from keras.utils import np_utils
import random
import torch
import torchvision.transforms as trans

class CIFAR10_Dataset:
    
    def __init__(self,
                 data_dir,
                 val_rate=0.1,
                 cutout = False):
        
        self.data_dir = data_dir
        self.val_rate = val_rate
        self.cutout = cutout
        
        self.load_dataset()
        self.split_train_and_val()
        
    
    
    def load_dataset(self):
        """
        Load all the data from CIFAR-10 dataset.
        """
        # Load meta information
        with open(os.path.join(self.data_dir, 'batches.meta'), 'rb') as f:
            label_names = pickle.load(f)['label_names']
            self.num_class = len(label_names) + 1
        
        # Load training data
        data = []
        labels = []
        for idx in range(1, 6):
            with open(os.path.join(self.data_dir, 'data_batch_%d'%(idx)), 'rb') as f:
                data_batch = pickle.load(f, encoding='bytes')
                data.append(data_batch[b'data'])
                labels.extend(data_batch[b'labels'])
        data = np.concatenate(data, axis=0)
        
        # Correlate data and labels
        imgs = []
        for idx in range(len(data)):
            imgs.append(np.reshape(data[idx],(3, 32, 32)) / 128 - 1)
        data = np.array(imgs)
        
        self.mean = [0.49139968, 0.48215827, 0.44653124]
        self.std = [0.24703233, 0.24348505, 0.26158768]

        #print(mean)
        #print(std)
        #self.preprocess = trans.Compose([trans.Normalize(mean, std),
        #                            trans.RandomCrop(size=32, padding=4),
        #                            trans.RandomHorizontalFlip(),
        #                            ])
        self.data_list = []
        for idx in range(len(labels)):
            # Normalize
            img = data[idx]
            for i in range(len(img)):
                img[i] = (img[i] - self.mean[i]) / self.std[i]
            #print(img)
            # Background = 0, object = 1 ~ 10
            label = labels[idx] + 1
            
            self.data_list.append({'image': img, 'label': label})

            
    def split_train_and_val(self):
        """
        Split part of training data into validation set.
        """
        random.shuffle(self.data_list)
        num_val = int(self.val_rate * len(self.data_list))
        # Split
        self.val_list = self.data_list[:num_val]
        self.train_list = self.data_list[num_val:]
        print ('-- Dataset Constructed')
        print ('-- Training set: %d samples' %(len(self.train_list)))
        print ('-- Validation set: %d samples' %(len(self.val_list)))
        print ('Cutout = ', self.cutout)
        
    def img_cutout(self, img):
        """
        cutout 3*32*32 img by patch size of 16
        """
        img = np.ascontiguousarray(img,dtype=np.float32)
        image = torch.from_numpy(img)
        
        b = random.randint(0, 32)
        c = random.randint(0, 32)
        #print(b,' ',c)
        if b < 8:
            if c < 8:
                image[:, 0:b+8, 0:c+8] = torch.zeros(3, b+8, c+8)
            elif c > 24:
                image[:, 0:b+8, c-8:32] = torch.zeros(3, b+8, 40-c)
            else:
                image[:, 0:b+8, c-8:c+8] = torch.zeros(3, b+8, 16)
        elif b > 24:
            if c < 8:
                image[:, b-8:32, 0:c+8] = torch.zeros(3, 40-b, c+8)
            elif c > 24:
                image[:, b-8:32, c-8:32] = torch.zeros(3, 40-b, 40-c)
            else:
                image[:, b-8:32, c-8:c+8] = torch.zeros(3, 40-b, 16)
        else:
            if c < 8:
                image[:, b-8:b+8, 0:c+8] = torch.zeros(3, 16, c+8)
            elif c > 24:
                image[:, b-8:b+8, c-8:32] = torch.zeros(3, 16, 40-c)
            else:
                image[:, b-8:b+8, c-8:c+8] = torch.zeros(3, 16, 16)
        
        return image.numpy()
    
    def batch_generator(self, batch_size=128, split='train'):
        """
        A generator to provide batch-wise data stream for training image classification task.
        """
        if split == 'train':
            data = self.train_list
        else:
            data = self.val_list
        
        # Record the traverse of data 
        data_idx = -1
        
        while True:
            # Construct batch data containers
            batch_img = np.zeros((batch_size, 3, 32, 32), np.float32)
            batch_gt = np.zeros((batch_size), np.float32) 
            
            i = 0
            while i < batch_size:
                data_idx = (data_idx + 1) % len(data)
                # Shuffle training data at the start of a new traverse
                if data_idx == 0 and split == 'train':
                    random.shuffle(data)
                    
                # Load data into batch
                this_img = data[data_idx]['image']
                
                #if split == 'train':
                # pad the img by 4
                this_img = np.pad(this_img, ((0,0),(4,4),(4,4)), mode='constant')
                # flip the img horizontally
                if random.randint(0,1) == 1:
                    this_img = np.flip(this_img, 2)
                # shake the img
                randx = random.randint(0,8)
                randy = random.randint(0,8)
                #print(randx, ' ' ,randy)
                this_img = this_img[:, randx:randx+32, randy:randy+32]
                if self.cutout:
                    this_img = self.img_cutout(this_img)
                
                batch_img[i] = this_img
                batch_gt[i] = data[data_idx]['label']
                i += 1
            
            yield batch_img, batch_gt
            
    def get_test_data(self):
        # Load test data
        with open(os.path.join(self.data_dir, 'test_batch'), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        
        datas = dict[b'data']
        labels = dict[b'labels']
        test_datas = []
        test_labels = []
        for idx in range(len(labels)):
            img =(datas[idx].reshape(3, 32, 32)/ 128 - 1)
            # Normalize
            for i in range(len(img)):
                img[i] = (img[i] - self.mean[i]) / self.std[i]
            #img = img.astype(np.float32)/128 - 1
            test_datas.append(img)
            # Background = 0, object = 1 ~ 10
            label = labels[idx] + 1
            test_labels.append(label)
        
        return test_datas, test_labels
                
if __name__ == '__main__':
    """
    Test the dataset module.
    """
    
    DATA_DIR = 'CIFAR10_dataset'
    VAL_RATE = 0.1
    
    dataset = CIFAR10_Dataset(data_dir=DATA_DIR,
                              val_rate=VAL_RATE)
    datagen = dataset.batch_generator()
    while True:
        data_batch = next(datagen)
        print (data_batch[0].shape, data_batch[1].shape)
