import numpy as np
import os
import h5py
#import gdal
import scipy.io as scio
#import pywt
import matplotlib.pyplot as plt
import cv2
def rotate_img(img, angle):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (0, 0)
    #获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    #计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    #调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img
def circshift(matrix,shiftnum1,shiftnum2):
    h,w,v=matrix.shape
    matrix[:shiftnum1, :, :] = 0
    matrix=np.vstack((matrix[shiftnum1:,:,:],matrix[:shiftnum1,:,:]))
    #matrix[:, (w - shiftnum2):, :] = 0
    matrix[:,:shiftnum2,:] = 0
    matrix=np.hstack((matrix[:,shiftnum2:,:],matrix[:,:shiftnum2,:]))
    return matrix
class DataSet(object):
    def __init__(self,pan_size,ms_size, hs_size, source_path,data_save_path,batch_size, stride, category='train'):
        self.pan_size=pan_size
        self.ms_size=ms_size
        self.hs_size = hs_size
        self.batch_size=batch_size
        if not os.path.exists(data_save_path):
            self.make_data(source_path,data_save_path,stride)
        self.pan,self.ms,self.hs,self.gt = self.read_data(data_save_path,category)
        #self.data_generator=self.generator()
        
    def generator(self):
        num_data=self.pan.shape[0]
        while True:
            #batch_pan=np.zeros((self.batch_size,self.pan_size,self.pan_size,3))
            batch_pan = np.zeros((self.batch_size, self.pan_size, self.pan_size, 3))
            batch_ms=np.zeros((self.batch_size,self.ms_size,self.ms_size,31))
            batch_gt = np.zeros((self.batch_size, self.pan_size, self.pan_size, 31))
            for i in range(self.batch_size):
                random_index=np.random.randint(0,num_data)
                batch_pan[i]=self.pan[random_index]
                batch_ms[i]=self.ms[random_index]
                batch_gt[i] = self.gt[random_index]
            yield batch_pan, batch_ms, batch_gt
    
    def read_data(self,path,category):
        f=h5py.File(path, 'r')
        if category == 'train':
            pan=np.array(f['pan_train'])
            #pan_edge = np.array(f['pan_edge_train'])
            ms=np.array(f['ms_train'])
            hs = np.array(f['hs_train'])
            gt = np.array(f['gt_train'])
        else:
            pan=np.array(f['pan_valid'])
            ms=np.array(f['ms_valid'])
            hs = np.array(f['hs_valid'])
            gt = np.array(f['gt_valid'])
        return pan, ms, hs, gt
        
    def make_data(self, source_path, data_save_path, stride):
        # source_ms_path=os.path.join(source_path, 'MS','1.TIF')
        # source_pan_path=os.path.join(source_path, 'Pan','1.TIF')
        for i in range(20):
            print(str(i))
            path = str(i + 1) + '.mat'
            source_hs_path = os.path.join(source_path, 'hs', path)
            source_ms_path = os.path.join(source_path, 'ms', path)
            source_pan_path = os.path.join(source_path, 'pan', path)
            source_gt_path = os.path.join(source_path, 'gt', path)
            pan = self.crop_to_patch(source_pan_path, stride, name='pan')
            ms = self.crop_to_patch(source_ms_path, stride, name='ms')
            hs = self.crop_to_patch(source_hs_path, stride, name='hs')
            #pan_edge = self.crop_to_patch(source_pan_path, stride, name='ms_edge')
            gt = self.crop_to_patch(source_gt_path, stride, name='gt')
            if i == 0:
                all_pan = pan
                all_ms = ms
                all_gt = gt
                all_hs = hs
                #all_pan_edge = pan_edge
            else:
                all_pan.extend(pan)
                all_ms.extend(ms)
                all_gt.extend(gt)
                all_hs.extend(hs)
                #all_pan_edge.extend(pan_edge)
        print('The number of ms patch is: ' + str(len(all_ms)))
        print('The number of pan patch is: ' + str(len(all_pan)))
        print('The number of gt patch is: ' + str(len(all_gt)))
        print('The number of hs patch is: ' + str(len(all_hs)))
        pan_train = all_pan
        ms_train = all_ms
        gt_train = all_gt
        hs_train = all_hs
        #pan_edge_train = all_pan_edge
        #pan_train, pan_valid, ms_train, ms_valid, gt_train, gt_valid = self.split_data(all_pan, all_ms, all_gt)
        #pan_train, pan_valid, ms_train, ms_valid, gt_train, gt_valid=self.split_data(all_pan,all_ms,all_gt)
        print('The number of pan_train patch is: ' + str(len(pan_train)))
        #print('The number of ms_valid patch is: ' + str(len(pan_valid)))
        print('The number of ms_train patch is: ' + str(len(ms_train)))
        print('The number of hs_train patch is: ' + str(len(hs_train)))
        #print('The number of hs_valid patch is: ' + str(len(ms_valid)))
        print('The number of gt_train patch is: ' + str(len(gt_train)))
        #print('The number of gt_valid patch is: ' + str(len(gt_valid)))
        pan_train=np.array(pan_train)
        #pan_valid=np.array(pan_valid)
        ms_train=np.array(ms_train)
        hs_train = np.array(hs_train)
        #ms_valid=np.array(ms_valid)
        gt_train = np.array(gt_train)
        #gt_valid = np.array(gt_valid)
        #pan_edge_train = np.array(pan_edge_train)
        f=h5py.File(data_save_path,'w')
        f.create_dataset('pan_train', data=pan_train)
        #f.create_dataset('pan_valid', data=pan_valid)
        f.create_dataset('ms_train', data=ms_train)
        f.create_dataset('hs_train', data=hs_train)

        f.create_dataset('gt_train', data=gt_train)


    def Normalization(x, Max, Min):
        x = (x - Min) / (Max - Min)
        return x

    from PIL import Image
    def crop_to_patch(self, img_path, stride, name):
        #img=(cv2.imread(img_path,-1)-127.5)/127.5
        img=self.read_img2(img_path)
        h=img.shape[0]
        w=img.shape[1]
        all_img=[]
        if name == 'ms':
            img = circshift(img, 1, 1)
            for i in range(0, h-self.ms_size+1, stride*4):
                for j in range(0, w-self.ms_size+1, stride*4):
                    img_patch=img[i:i+self.ms_size, j:j+self.ms_size,:]
                    all_img.append(img_patch)
        elif name == 'gt':
            for i in range(0, h-self.pan_size+1, stride*16):
                for j in range(0, w-self.pan_size+1, stride*16):
                    img_patch=img[i:i+self.pan_size, j:j+self.pan_size, :]
                    all_img.append(img_patch)
        elif name == 'pan':
            #img = edge_detected(img)
            img = np.expand_dims(img, axis=2)
            for i in range(0, h-self.pan_size+1, stride*16):
                for j in range(0, w-self.pan_size+1, stride*16):
                    img_patch=img[i:i+self.pan_size, j:j+self.pan_size, :]
                    all_img.append(img_patch)
        else:
            for i in range(0, h-self.hs_size+1, stride):
                for j in range(0, w-self.hs_size+1, stride):
                    img_patch=img[i:i+self.hs_size, j:j+self.hs_size, :]
                    all_img.append(img_patch)
        return all_img

    def split_data(self,all_pan,all_ms,all_gt):
        ''' all_pan和all_ms均为list'''
        pan_train=[]
        pan_valid=[]
        ms_train=[]
        ms_valid=[]
        gt_train = []
        gt_valid = []
        for i in range(len(all_pan)):
            rand=np.random.randint(0,100)
            if rand <=10:
                pan_valid.append(all_pan[i])
                ms_valid.append(all_ms[i])
                gt_valid.append(all_gt[i])
            else:
                ms_train.append(all_ms[i])
                pan_train.append(all_pan[i])
                gt_train.append(all_gt[i])
        return pan_train, pan_valid, ms_train, ms_valid, gt_train, gt_valid
        
    def read_img(self,path,name):
        data=gdal.Open(path)
        w=data.RasterXSize
        h=data.RasterYSize
        img=data.ReadAsArray(0,0,w,h)
        if name == 'ms':
            img=np.transpose(img,(1,2,0))
        img=(img-1023.5)/1023.5
        return img
        
    def read_img2(self, path):
        img=scio.loadmat(path)['I']
        #img=(img-127.5)/127.5
        return img

    '''
            import cv2
            db1 = pywt.Wavelet('db1')
            for i in range(np.size(img, 2)):
                img1 = img[:,:,i]
                cA1, cA2, cA3 = pywt.swtn(img1, db1, level=3)
                #data1 = cA1['aa']
                data2 = cA1['ad']
                data3 = cA1['da']
                data4 = cA1['dd']

                #data5 = cA2['aa']
                data6 = cA2['ad']
                data7 = cA2['da']
                data8 = cA2['dd']

                data9 = cA3['aa']
                data10 = cA3['ad']
                data11 = cA3['da']
                data12 = cA3['dd']

                data9 = 255 * (data9-np.min(data9)) / (np.max(data9)-np.min(data9))
                cv2.imwrite("wavelet\\"+ str(i)+"\\c3.jpg", data9)

                data2 = 255 * (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
                cv2.imwrite("wavelet\\" + str(i) + "\\w11.jpg", data2)

                data3 = 255 * (data3 - np.min(data3)) / (np.max(data3) - np.min(data3))
                cv2.imwrite("wavelet\\" + str(i) + "\\w21.jpg", data3)

                data4 = 255 * (data4 - np.min(data4)) / (np.max(data4) - np.min(data4))
                cv2.imwrite("wavelet\\" + str(i) + "\\w31.jpg", data4)

                data6 = 255 * (data6 - np.min(data6)) / (np.max(data6) - np.min(data6))
                cv2.imwrite("wavelet\\" + str(i) + "\\w12.jpg", data6)

                data7 = 255 * (data7 - np.min(data7)) / (np.max(data7) - np.min(data7))
                cv2.imwrite("wavelet\\" + str(i) + "\\w22.jpg", data7)

                data8 = 255 * (data8 - np.min(data8)) / (np.max(data8) - np.min(data8))
                cv2.imwrite("wavelet\\" + str(i) + "\\w32.jpg", data8)

                data10 = 255 * (data10 - np.min(data10)) / (np.max(data10) - np.min(data10))
                cv2.imwrite("wavelet\\" + str(i) + "\\w13.jpg", data10)

                data11 = 255 * (data11 - np.min(data11)) / (np.max(data11) - np.min(data11))
                cv2.imwrite("wavelet\\" + str(i) + "\\w23.jpg", data11)

                data12 = 255 * (data12 - np.min(data12)) / (np.max(data12) - np.min(data12))
                cv2.imwrite("wavelet\\" + str(i) + "\\w33.jpg", data12)
    '''
      
        
                    
         
    
    
 
                
