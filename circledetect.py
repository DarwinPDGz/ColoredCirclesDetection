import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
# import pandas as pd
# import openpyxl as op

class CircleDetect:
    COLOR_LIST = [
        'red',
        'green',
        'blue',
        'cyan',
        'magenta',
        'yellow',
        'black']
    
    SET_ORDER_F_SITA = {
        'black' : 0,
        'red': 2,
        'yellow': 1,
        'blue': 3}

    SET_ORDER_B_SITA = {
        'black': 0,
        'cyan': 3,
        'magenta': 2,
        'green': 1}
    
    SET_ORDER_F_RAWANA = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_RAWANA = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_RAMA = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_RAMA = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_HANUMAN = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_HANUMAN = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_LAKSMANA = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_LAKSMANA = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_SUGRIWA = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_SUGRIWA = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_BALI = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_BALI = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_WIBHISANA = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_WIBHISANA = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_ANGADA = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_ANGADA = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
    
    SET_ORDER_F_ANILA = {
        'black' : 0,
        'red': 1,
        'green': 2,
        'blue': 3}

    SET_ORDER_B_ANILA = {
        'black': 0,
        'cyan': 1,
        'magenta': 2,
        'yellow': 3}
      
    def __init__(self, image_path, cct=0, range_type=0, arrayVersion=False, thresholdVersion=False, detectFULL=False, minimumDist=20, parameter1=70, parameter2=20, minimumRadius=0, maximumRadius=30, tolerance=5, ptpthreshold=15, set_order_f=SET_ORDER_F_SITA, set_order_b=SET_ORDER_B_SITA):
        self.color_array = CircleDetect.setting_color_array(arrayVersion)
        self.image_path = image_path
        self.minimumDist = minimumDist
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.minimumRadius = minimumRadius
        self.maximumRadius = maximumRadius
        self.tolerance = tolerance
        self.ptpthreshold = ptpthreshold
        self.set_order_f = set_order_f
        self.set_order_b = set_order_b
        self.color_check_type = cct
        self.range_type = range_type
        
        img = cv2.imread(str(self.image_path))
        assert img is not None, "file could not be read, check with os.path.exists()"
        
        cimg = img.copy()
        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
        imgr = cimg[:,:,0]
        imgg = cimg[:,:,1]
        imgb = cimg[:,:,2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.medianBlur(img,11)
        imgr = cv2.medianBlur(imgr,11)
        imgg = cv2.medianBlur(imgg,11)
        imgb = cv2.medianBlur(imgb,11)
        
        img = cv2.resize(img, (0,0), img, 0.2, 0.2)
        cimg = cv2.resize(cimg, (0,0), cimg, 0.2, 0.2)
        imgr = cv2.resize(imgr, (0,0), imgr, 0.2, 0.2)
        imgg = cv2.resize(imgg, (0,0), imgg, 0.2, 0.2)
        imgb = cv2.resize(imgb, (0,0), imgb, 0.2, 0.2)
        
        # invimg = cv2.bitwise_not(img)
        # self.iimg = invimg
        
        self.img, self.cimg, self.imgr, self.imgg, self.imgb = img, cimg, imgr, imgg, imgb
        
        if thresholdVersion:
            CircleDetect.threshold_img(self)
        
        self.circles = CircleDetect.detect_circles(self, self.img)
        self.circlesr = CircleDetect.detect_circles(self, self.imgr)
        self.circlesg = CircleDetect.detect_circles(self, self.imgg)
        self.circlesb = CircleDetect.detect_circles(self, self.imgb)
        # self.circlesi = CircleDetect.detect_circles(self, self.iimg)
        
        # self.combined_circles = np.concatenate((self.circles, self.circlesr, self.circlesg, self.circlesb, self.circlesi), axis=1)
        self.combined_circles = np.concatenate((self.circles, self.circlesr, self.circlesg, self.circlesb), axis=1)
        
        if detectFULL:
            self.circles = self.combined_circles
        
        self.filtered_circles = CircleDetect.filter_circles(self)
        self.filtered_circles2 = self.filtered_circles
        
        print(self.filtered_circles.shape)
        
    def detect_circles(self, img):
        
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,self.minimumDist,
                            param1=self.parameter1,param2=self.parameter2,minRadius=self.minimumRadius,maxRadius=self.maximumRadius)
        circles = np.int32(np.around(circles))
        return circles    
        
    def setting_color_array(arrayVersion):
        color_array = np.zeros(shape=(3,len(CircleDetect.COLOR_LIST)))
        if arrayVersion:
            for i, color in enumerate(CircleDetect.COLOR_LIST):
                if color == 'red':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 142, 35, 45
                elif color == 'green':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 96, 126, 64
                elif color == 'blue':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 39, 51, 99
                elif color == 'cyan':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 117, 164, 174
                elif color == 'magenta':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 126, 49, 101
                elif color == 'yellow':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 212, 184, 41
                elif color == 'black':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 30, 29, 35
                else:
                    continue
                    # print('color not found')
        else:
            for i, color in enumerate(CircleDetect.COLOR_LIST):
                if color == 'red':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 150, 17, 34
                elif color == 'green':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 98, 119, 44
                elif color == 'blue':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 17, 23, 108
                elif color == 'cyan':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 118, 161, 180
                elif color == 'magenta':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 143, 49, 133
                elif color == 'yellow':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 196, 171, 17
                elif color == 'black':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 22, 23, 25
                else:
                    continue
                    # print('color not found')
            
        color_array = color_array.astype(np.uint8)
        return color_array 
    
    def get_lab_range(self, circleposx, circleposy, radius, img, tolerance=8):
        
        lab_cimg = cv2.cvtColor(self.cimg, cv2.COLOR_RGB2LAB)
        
        imgl = lab_cimg[:,:,0]
        imga = lab_cimg[:,:,1]
        imgb = lab_cimg[:,:,2]
        
        if radius < tolerance:
            return 0, 0
        
        mask = np.zeros_like(img)
        cv2.circle(mask, (circleposx, circleposy), (radius-tolerance), 255, -1)
        # masked_img = cv2.bitwise_and(img, mask)
        masked_imgl = cv2.bitwise_and(imgl, mask)
        masked_imga = cv2.bitwise_and(imga, mask)
        masked_imgb = cv2.bitwise_and(imgb, mask)
        
        num = (img.shape[0])*(img.shape[1])
        
        # masked_reshape = masked_img.reshape(1,num)
        masked_reshape_l = masked_imgl.reshape(1,num)
        masked_reshape_a = masked_imga.reshape(1,num)
        masked_reshape_b = masked_imgb.reshape(1,num)
        reshape = mask.reshape(1,num)
        
        new_list_l = []
        new_list_a = []
        new_list_b = []

        for i in range(int(num)):
            if reshape[0][i] == 255:
                # new_list.append(masked_reshape[0][i])
                new_list_l.append(masked_reshape_l[0][i])
                new_list_a.append(masked_reshape_a[0][i])
                new_list_b.append(masked_reshape_b[0][i])
                
        new_list_l = np.array(new_list_l)
        new_list_a = np.array(new_list_a)
        new_list_b = np.array(new_list_b)
                
        # grab the most volatile values
        new_list = []
        ptp_l = np.ptp(new_list_l)   
        ptp_a = np.ptp(new_list_a)
        ptp_b = np.ptp(new_list_b)
        
        new_list.append(ptp_l)
        new_list.append(ptp_a)
        new_list.append(ptp_b)
        
        # print(new_list)
        
        new_list = np.array(new_list)
        
        return new_list[0], new_list[1], new_list[2]
    
    def lex_sorter(self, circleposx, circleposy, radius, img):
        cimg = self.cimg
        cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2HLS)
        
        imgh = cimg[:,:,0]
        imgl = cimg[:,:,1]
        imgs = cimg[:,:,2]
        
        mask = np.zeros_like(img)
        cv2.circle(mask, (circleposx, circleposy), radius, 255, -1)
        masked_imgh = cv2.bitwise_and(imgh, mask)
        masked_imgl = cv2.bitwise_and(imgl, mask)
        masked_imgs = cv2.bitwise_and(imgs, mask)
        
        num = (img.shape[0])*(img.shape[1])
        
        masked_reshape_h = masked_imgh.reshape(1,num)
        masked_reshape_l = masked_imgl.reshape(1,num)
        masked_reshape_s = masked_imgs.reshape(1,num)
        reshape = mask.reshape(1,num)
        
        new_list_h = []
        new_list_l = []
        new_list_s = []
        
        for i in range(int(num)):
            if reshape[0][i] == 255:
                new_list_h.append(masked_reshape_h[0][i])
                new_list_l.append(masked_reshape_l[0][i])
                new_list_s.append(masked_reshape_s[0][i])
                
        new_list_h = np.array(new_list_h)
        new_list_l = np.array(new_list_l)
        new_list_s = np.array(new_list_s)
        
        sorted_h = np.sort(new_list_h)
        sorted_l = np.sort(new_list_l)
        sorted_s = np.sort(new_list_s)
        
        return sorted_h, sorted_l, sorted_s
    
    def hls_visualizer(self, circleposx, circleposy, radius, img):
        h, l, s = CircleDetect.lex_sorter(self, circleposx, circleposy, radius, img)
        
        h_strip = np.zeros((20,len(h),3), dtype=np.uint8)
        l_strip = np.zeros((20,len(l),3), dtype=np.uint8)
        s_strip = np.zeros((20,len(s),3), dtype=np.uint8)
        
        print(h_strip.shape)
        print(l_strip.shape)
        print(s_strip.shape)
        
        for j in range(20):
            for i in range(len(h)):
                print(i)
                # convert hue into color wheel/strip
                h_strip[j,i] = [h[i], 127, 255]
                l_strip[j,i] = [0, l[i], 255]
                s_strip[j,i] = [0, 127, s[i]]
            
        h_strip = cv2.cvtColor(h_strip, cv2.COLOR_HLS2RGB)
        l_strip = cv2.cvtColor(l_strip, cv2.COLOR_HLS2RGB)
        s_strip = cv2.cvtColor(s_strip, cv2.COLOR_HLS2RGB)
        
        fig = plt.figure(figsize=(20,20))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        
        # plt.subplot(1,3,1)
        # plt.imshow(h_strip)
        # plt.subplot(1,3,2)
        # plt.imshow(l_strip)
        # plt.subplot(1,3,3)
        # plt.imshow(s_strip)
        # plt.show()
            
    def get_range(self, circleposx, circleposy, radius, img, tolerance=8):
        
        imgr = self.cimg[:,:,0]
        imgg = self.cimg[:,:,1]
        imgb = self.cimg[:,:,2]
        
        if radius < tolerance:
            return 0, 0
        
        mask = np.zeros_like(img)
        cv2.circle(mask, (circleposx, circleposy), (radius-tolerance), 255, -1)
        # masked_img = cv2.bitwise_and(img, mask)
        masked_imgr = cv2.bitwise_and(imgr, mask)
        masked_imgg = cv2.bitwise_and(imgg, mask)
        masked_imgb = cv2.bitwise_and(imgb, mask)
        
        num = (img.shape[0])*(img.shape[1])
        
        # masked_reshape = masked_img.reshape(1,num)
        masked_reshape_r = masked_imgr.reshape(1,num)
        masked_reshape_g = masked_imgg.reshape(1,num)
        masked_reshape_b = masked_imgb.reshape(1,num)
        reshape = mask.reshape(1,num)
        
        new_list_r = []
        new_list_g = []
        new_list_b = []

        for i in range(int(num)):
            if reshape[0][i] == 255:
                # new_list.append(masked_reshape[0][i])
                new_list_r.append(masked_reshape_r[0][i])
                new_list_g.append(masked_reshape_g[0][i])
                new_list_b.append(masked_reshape_b[0][i])
                
        new_list_r = np.array(new_list_r)
        new_list_g = np.array(new_list_g)
        new_list_b = np.array(new_list_b)
                
        # grab the most volatile values
        new_list = []
        ptp_r = np.ptp(new_list_r)   
        ptp_g = np.ptp(new_list_g)
        ptp_b = np.ptp(new_list_b)
        
        new_list.append(ptp_r)
        new_list.append(ptp_g)
        new_list.append(ptp_b)
        
        # print(new_list)
        
        new_list = np.array(new_list)
        
        return new_list[0], new_list[1], new_list[2]
    
    def dE00_comparator(self, circleposx, circleposy, radius, cimg):
        pass
    
    def dE94_comparator(self, circleposx, circleposy, radius, cimg):
        color_array = self.color_array
    
        list_of_colors = np.zeros((1,7,3), dtype=np.uint8)
        
        lab_cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2LAB)
        
        for i in range(len(color_array[0])):
            
            list_of_colors[0][i] = color_array[0][i], color_array[1][i], color_array[2][i]
            
        list_of_colors = cv2.cvtColor(list_of_colors, cv2.COLOR_RGB2LAB)
        
        for i in range(len(list_of_colors[0])):
            a = list_of_colors[0][i][1] 
            b = list_of_colors[0][i][2]
            
            c = math.sqrt(a**2 + b**2)
            h = math.atan2(b, a)
            
            list_of_colors[0][i][1] = c
            list_of_colors[0][i][2] = h
            
        # list of colors is now in LCH
        
        grayscale = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(grayscale)
        cv2.circle(mask, (circleposx, circleposy), (radius-self.tolerance), 255, -1)
        
        masked_cimg_L = cv2.bitwise_and(lab_cimg[:,:,0], mask)
        masked_cimg_A = cv2.bitwise_and(lab_cimg[:,:,1], mask)
        masked_cimg_B = cv2.bitwise_and(lab_cimg[:,:,2], mask)
        
        num = (cimg.shape[0])*(cimg.shape[1])
        
        masked_reshape_L = masked_cimg_L.reshape(1,num)
        masked_reshape_A = masked_cimg_A.reshape(1,num)
        masked_reshape_B = masked_cimg_B.reshape(1,num)
        
        reshape = mask.reshape(1,num)
        
        new_list_L = []
        new_list_A = []
        new_list_B = []

        for i in range(int(num)):
            if reshape[0][i] == 255:
                new_list_L.append(masked_reshape_L[0][i])
                new_list_A.append(masked_reshape_A[0][i])
                new_list_B.append(masked_reshape_B[0][i])
                
        new_list_L = np.array(new_list_L)
        new_list_A = np.array(new_list_A)
        new_list_B = np.array(new_list_B)
        
        mean_L = np.mean(new_list_L)
        mean_A = np.mean(new_list_A)
        mean_B = np.mean(new_list_B)
        
        # convert to LCH
        
        mean_C = math.sqrt(mean_A**2 + mean_B**2)
        mean_H = math.atan2(mean_B, mean_A)
        
        delta_E_array = []
        
        # deltaE94 
        for i in range(len(list_of_colors[0])):
            # delta_E = math.sqrt((mean_L - list_of_colors[0][i][0])**2 + (mean_C - list_of_colors[0][i][1])**2 + (mean_H - list_of_colors[0][i][2])**2) 
            delta_E = math.sqrt((mean_L - list_of_colors[0][i][0])**2 + ((mean_C - list_of_colors[0][i][1])**2)/(1+0.045) + ((mean_H - list_of_colors[0][i][2])**2)/(1+0.015)) 
            # delta_E = (mean_L - list_of_colors[0][i][0])**2 + (mean_C - list_of_colors[0][i][1])**2 + (mean_H - list_of_colors[0][i][2])**2

            # print(delta_E)      
            delta_E_array.append(delta_E)
            
        delta_E_array = np.array(delta_E_array)
        determined_color = ''
        
        if delta_E_array.min() == delta_E_array[0]:
            determined_color = 'red'
            # print('red')
        elif delta_E_array.min() == delta_E_array[1]:
            determined_color = 'green'
            # print('green')
        elif delta_E_array.min() == delta_E_array[2]:
            determined_color = 'blue'
            # print('blue')
        elif delta_E_array.min() == delta_E_array[3]:
            determined_color = 'cyan'
            # print('cyan')
        elif delta_E_array.min() == delta_E_array[4]:
            determined_color = 'magenta'
            # print('magenta')
        elif delta_E_array.min() == delta_E_array[5]:
            determined_color = 'yellow'
            # print('yellow')
        elif delta_E_array.min() == delta_E_array[6]:
            determined_color = 'black'
            # print('black')
        else:
            pass
            # print('color not found')
        return determined_color, delta_E_array.min()

    def dE76_comparator(self, circleposx, circleposy, radius, cimg):
        color_array = self.color_array
        
        list_of_colors = np.zeros((1,7,3), dtype=np.uint8)
        
        lab_cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2LAB)
        
        for i in range(len(color_array[0])):
            list_of_colors[0][i] = color_array[0][i], color_array[1][i], color_array[2][i]
            
        list_of_colors = cv2.cvtColor(list_of_colors, cv2.COLOR_RGB2LAB)
        
        grayscale = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(grayscale)
        cv2.circle(mask, (circleposx, circleposy), (radius-self.tolerance), 255, -1)
        
        masked_cimg_L = cv2.bitwise_and(lab_cimg[:,:,0], mask)
        masked_cimg_A = cv2.bitwise_and(lab_cimg[:,:,1], mask)
        masked_cimg_B = cv2.bitwise_and(lab_cimg[:,:,2], mask)
        
        num = (cimg.shape[0])*(cimg.shape[1])
        
        masked_reshape_L = masked_cimg_L.reshape(1,num)
        masked_reshape_A = masked_cimg_A.reshape(1,num)
        masked_reshape_B = masked_cimg_B.reshape(1,num)
        
        reshape = mask.reshape(1,num)
        
        new_list_L = []
        new_list_A = []
        new_list_B = []

        for i in range(int(num)):
            if reshape[0][i] == 255:
                new_list_L.append(masked_reshape_L[0][i])
                new_list_A.append(masked_reshape_A[0][i])
                new_list_B.append(masked_reshape_B[0][i])
                
        new_list_L = np.array(new_list_L)
        new_list_A = np.array(new_list_A)
        new_list_B = np.array(new_list_B)
        
        mean_L = np.mean(new_list_L)
        mean_A = np.mean(new_list_A)
        mean_B = np.mean(new_list_B)
        
        delta_E_array = []
        
        for i in range(len(list_of_colors[0])):
            delta_E = math.sqrt((mean_L - list_of_colors[0][i][0])**2 + (mean_A - list_of_colors[0][i][1])**2 + (mean_B - list_of_colors[0][i][2])**2) 
            # print(delta_E)
            delta_E_array.append(delta_E)
            
        delta_E_array = np.array(delta_E_array)
        determined_color = ''
        
        if delta_E_array.min() == delta_E_array[0]:
            determined_color = 'red'
            # print('red')
        elif delta_E_array.min() == delta_E_array[1]:
            determined_color = 'green'
            # print('green')
        elif delta_E_array.min() == delta_E_array[2]:
            determined_color = 'blue'
            # print('blue')
        elif delta_E_array.min() == delta_E_array[3]:
            determined_color = 'cyan'
            # print('cyan')
        elif delta_E_array.min() == delta_E_array[4]:
            determined_color = 'magenta'
            # print('magenta')
        elif delta_E_array.min() == delta_E_array[5]:
            determined_color = 'yellow'
            # print('yellow')
        elif delta_E_array.min() == delta_E_array[6]:
            determined_color = 'black'
            # print('black')
        else:
            pass
            # print('color not found')
        return determined_color, delta_E_array.min()    
    
    def rgb_comparator(self, circleposx, circleposy, radius, cimg):
        color_array = self.color_array
        
        grayscale = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(grayscale)
        cv2.circle(mask, (circleposx, circleposy), (radius-self.tolerance), 255, -1)
        masked_cimg_r = cv2.bitwise_and(cimg[:,:,0], mask)
        masked_cimg_g = cv2.bitwise_and(cimg[:,:,1], mask)
        masked_cimg_b = cv2.bitwise_and(cimg[:,:,2], mask)
        
        num = (cimg.shape[0])*(cimg.shape[1])
        
        masked_reshape_r = masked_cimg_r.reshape(1,num)
        masked_reshape_g = masked_cimg_g.reshape(1,num)
        masked_reshape_b = masked_cimg_b.reshape(1,num)
        
        reshape = mask.reshape(1,num)
        
        new_list_r = []
        new_list_g = []
        new_list_b = []

        for i in range(int(num)):
            if reshape[0][i] == 255:
                new_list_r.append(masked_reshape_r[0][i])
                new_list_g.append(masked_reshape_g[0][i])
                new_list_b.append(masked_reshape_b[0][i])
                
        new_list_r = np.array(new_list_r)
        new_list_g = np.array(new_list_g)
        new_list_b = np.array(new_list_b)
        
        mean_r = np.mean(new_list_r)
        mean_g = np.mean(new_list_g)
        mean_b = np.mean(new_list_b)
        
        delta_rgb_array = []
        
        # print(len(color_array[0]))
        
        for i in range(len(color_array[0])):
            delta_rgb = math.sqrt((mean_r - color_array[0][i])**2 + (mean_g - color_array[1][i])**2 + (mean_b - color_array[2][i])**2)
            # print(delta_E)
            delta_rgb_array.append(delta_rgb)
            
        delta_rgb_array = np.array(delta_rgb_array)
        determined_color = ''
        
        if delta_rgb_array.min() == delta_rgb_array[0]:
            determined_color = 'red'
            # print('red')
        elif delta_rgb_array.min() == delta_rgb_array[1]:
            determined_color = 'green'
            # print('green')
        elif delta_rgb_array.min() == delta_rgb_array[2]:
            determined_color = 'blue'
            # print('blue')
        elif delta_rgb_array.min() == delta_rgb_array[3]:
            determined_color = 'cyan'
            # print('cyan')
        elif delta_rgb_array.min() == delta_rgb_array[4]:
            determined_color = 'magenta'
            # print('magenta')
        elif delta_rgb_array.min() == delta_rgb_array[5]:
            determined_color = 'yellow'
            # print('yellow')
        elif delta_rgb_array.min() == delta_rgb_array[6]:
            determined_color = 'black'
            # print('black')
        else:
            pass
            # print('color not found')
        return determined_color, delta_rgb_array.min()
    
    def alt_filter_circles(self):
        # hue sorter, lexicographical order
        
        pass
     
    def filter_circles(self):
        # img, cimg, circles = CircleDetect.get_img_and_circles(self)
        circles = self.circles
        
        list_of_ranges = []
        list_of_ranges2 = []
        list_of_ranges3 = []

        # print(mean, range)
        if self.color_check_type == 0:
            for i in range(self.circles.shape[1]):
                ranges, ranges2, ranges3  = CircleDetect.get_range(self, circles[0][i][0], circles[0][i][1], circles[0][i][2], self.img, tolerance=self.tolerance)
                list_of_ranges.append(ranges)
                list_of_ranges2.append(ranges2)
                list_of_ranges3.append(ranges3)
                # print(mean, ranges)
        else:
            for i in range(self.circles.shape[1]):
                ranges, ranges2, ranges3 = CircleDetect.get_lab_range(self, circles[0][i][0], circles[0][i][1], circles[0][i][2], self.img, tolerance=self.tolerance)
                list_of_ranges.append(ranges)
                list_of_ranges2.append(ranges2)
                list_of_ranges3.append(ranges3)
                # print(mean, ranges)
            
        delete_list = []

        for i in range(len(list_of_ranges)):
            if list_of_ranges[i] > self.ptpthreshold or list_of_ranges2[i] > self.ptpthreshold or list_of_ranges3[i] > self.ptpthreshold:
                print(i)
                delete_list.append(i)
                # circles = np.delete(circles, i, axis=1)
                
        delete_list = sorted(delete_list, reverse=True)

        for i in delete_list:
            circles = np.delete(circles, i, axis=1)
            
        filtered_circles = circles
            
        return filtered_circles
        
    def detect_colors(self, type=0):
        filtered_circles = self.filtered_circles
        
        recognized_colors = []
        instability_factor = []
        
        if type == 0:
            print("rgb")
            for i in range(filtered_circles.shape[1]):
                
                rcol, ifac = CircleDetect.rgb_comparator(self, filtered_circles[0][i][0], filtered_circles[0][i][1], filtered_circles[0][i][2], self.cimg)
                
                recognized_colors.append(rcol)
                instability_factor.append(ifac)
        elif type==1:
            print("dE76")
            for i in range(filtered_circles.shape[1]):
                
                rcol, ifac = CircleDetect.dE76_comparator(self, filtered_circles[0][i][0], filtered_circles[0][i][1], filtered_circles[0][i][2], self.cimg)
                
                recognized_colors.append(rcol)
                instability_factor.append(ifac)
        elif type==2:
            print("dE94")
            for i in range(filtered_circles.shape[1]):
                rcol, ifac = CircleDetect.dE94_comparator(self, filtered_circles[0][i][0], filtered_circles[0][i][1], filtered_circles[0][i][2], self.cimg)
                
                recognized_colors.append(rcol)
                instability_factor.append(ifac)
        elif type==3:
            print("dE00")
            for i in range(filtered_circles.shape[1]):
                rcol, ifac = CircleDetect.dE00_comparator(self, filtered_circles[0][i][0], filtered_circles[0][i][1], filtered_circles[0][i][2], self.cimg)
                
                recognized_colors.append(rcol)
                instability_factor.append(ifac)
        else:
            return 0, 0
            # print('no type selected')
            
        
        return recognized_colors, instability_factor
    
    def anti_duplicates(self):
        # fix this to compare HSL, not RGB (make a dE94 comparator)
        
        filtered_circles = self.filtered_circles
        seen_colors, instability_factor = CircleDetect.detect_colors(self, self.range_type)
        
        print(filtered_circles)
        print(filtered_circles.shape)
        
        true_filtered_circles = np.zeros((1,7,3), dtype=np.int32)
        true_seen_colors = []
        
        red_list = []
        green_list = []
        blue_list = []
        cyan_list = []
        magenta_list = []
        yellow_list = []
        black_list = []
        
        color_counter = np.zeros((1,7), dtype=np.int32)
        
        for i in range(len(seen_colors)):
            if seen_colors[i] == 'red':
                red_list.append(i)
                color_counter[0][0] += 1
            elif seen_colors[i] == 'green':
                green_list.append(i)
                color_counter[0][1] += 1
            elif seen_colors[i] == 'blue':
                blue_list.append(i)
                color_counter[0][2] += 1
            elif seen_colors[i] == 'cyan':
                cyan_list.append(i)
                color_counter[0][3] += 1
            elif seen_colors[i] == 'magenta':
                magenta_list.append(i)
                color_counter[0][4] += 1
            elif seen_colors[i] == 'yellow':
                yellow_list.append(i)
                color_counter[0][5] += 1
            elif seen_colors[i] == 'black':
                black_list.append(i)
                color_counter[0][6] += 1
            else:
                continue
                # print('color not found/not in set')       
                
        print(color_counter)
        print(seen_colors)
        print('ifac', instability_factor)
        print('red', red_list)
        for i in red_list:
            print(instability_factor[i])
        print('green', green_list)
        for i in green_list:
            print(instability_factor[i])
        print('blue', blue_list)
        for i in blue_list:
            print(instability_factor[i])
        print('cyan', cyan_list)
        for i in cyan_list:
            print(instability_factor[i])
        print('magenta', magenta_list)
        for i in magenta_list:
            print(instability_factor[i])
        print('yellow', yellow_list)
        for i in yellow_list:
            print(instability_factor[i])
        print('black', black_list)
        for i in black_list:
            print(instability_factor[i])
                
        for i in range(len(color_counter[0])):
            # choosing the most similar color
            current_color = np.zeros((1,3), dtype=np.int32)
            if color_counter[0][i] > 1:
                if i == 0:
                    for j in range(len(red_list)):
                        if j == 0:
                            current_color = filtered_circles[0][red_list[j]]
                        else:
                            if instability_factor[red_list[j]] > instability_factor[red_list[j-1]]:
                                continue
                            else:
                                current_color = filtered_circles[0][red_list[j]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('red')
                elif i == 1:
                    for j in range(len(green_list)):
                        if j == 0:
                            current_color = filtered_circles[0][green_list[j]]
                        else:
                            if instability_factor[green_list[j]] > instability_factor[green_list[j-1]]:
                                continue
                            else:
                                current_color = filtered_circles[0][green_list[j]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('green')
                elif i == 2:
                    for j in range(len(blue_list)):
                        if j == 0:
                            current_color = filtered_circles[0][blue_list[j]]
                        else:
                            if instability_factor[blue_list[j]] > instability_factor[blue_list[j-1]]:
                                continue
                            else:
                                current_color = filtered_circles[0][blue_list[j]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('blue')
                elif i == 3:
                    for j in range(len(cyan_list)):
                        if j == 0:
                            current_color = filtered_circles[0][cyan_list[j]]
                        else:
                            if instability_factor[cyan_list[j]] > instability_factor[cyan_list[j-1]]:
                                continue
                            else:
                                current_color = filtered_circles[0][cyan_list[j]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('cyan')
                elif i == 4:
                    for j in range(len(magenta_list)):
                        if j == 0:
                            current_color = filtered_circles[0][magenta_list[j]]
                        else:
                            if instability_factor[magenta_list[j]] > instability_factor[magenta_list[j-1]]:
                                continue
                            else:
                                current_color = filtered_circles[0][magenta_list[j]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('magenta')
                elif i == 5:
                    for j in range(len(yellow_list)):
                        if j == 0:
                            current_color = filtered_circles[0][yellow_list[j]]
                        else:
                            if instability_factor[yellow_list[j]] > instability_factor[yellow_list[j-1]]:
                                continue
                            else:
                                current_color = filtered_circles[0][yellow_list[j]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('yellow')
                elif i == 6:
                    for j in range(len(black_list)):
                        if j == 0:
                            current_color = filtered_circles[0][black_list[j]]
                        else:
                            if instability_factor[black_list[j]] > instability_factor[black_list[j-1]]:
                                continue
                            else:
                                current_color = filtered_circles[0][black_list[j]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('black')

            else:
                if i == 0:
                    current_color = filtered_circles[0][red_list[0]]
                    true_filtered_circles[0][i] = current_color 
                    true_seen_colors.append('red')
                elif i == 1:
                    current_color = filtered_circles[0][green_list[0]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('green')
                elif i == 2:
                    current_color = filtered_circles[0][blue_list[0]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('blue')
                elif i == 3:
                    current_color = filtered_circles[0][cyan_list[0]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('cyan')
                elif i == 4:
                    current_color = filtered_circles[0][magenta_list[0]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('magenta')
                elif i == 5:
                    current_color = filtered_circles[0][yellow_list[0]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('yellow')
                elif i == 6:
                    current_color = filtered_circles[0][black_list[0]]
                    true_filtered_circles[0][i] = current_color
                    true_seen_colors.append('black')
                else:
                    continue
                    # print('color not found/not in set')
                    
        print('\n')
        print(true_seen_colors)
        print(true_filtered_circles)
                                
        return true_filtered_circles, true_seen_colors
    
    def reorder_circles(self):
        self.filtered_circles2, recognized_colors = CircleDetect.anti_duplicates(self)

        reordered_circles_f = np.zeros((1,4,3), dtype=np.int32)
        reordered_circles_b = np.zeros((1,4,3), dtype=np.int32)
        
        reordered_list_f = []
        reordered_list_b = []
        
        reordered_list_f = list(self.set_order_f.keys())
        reordered_list_b = list(self.set_order_b.keys())

        for i in range(len(recognized_colors)):
            if recognized_colors[i] == reordered_list_f[0]:
                reordered_circles_f[0][self.set_order_f[recognized_colors[i]]] = self.filtered_circles2[0][i]
            elif recognized_colors[i] == reordered_list_f[1]:
                reordered_circles_f[0][self.set_order_f[recognized_colors[i]]] = self.filtered_circles2[0][i]
            elif recognized_colors[i] == reordered_list_f[2]:
                reordered_circles_f[0][self.set_order_f[recognized_colors[i]]] = self.filtered_circles2[0][i]
            elif recognized_colors[i] == reordered_list_f[3]:
                reordered_circles_f[0][self.set_order_f[recognized_colors[i]]] = self.filtered_circles2[0][i]
            else:
                continue
                # print('color not found/not in set')
                
        for i in range(len(recognized_colors)):
            if recognized_colors[i] == reordered_list_b[0]:
                reordered_circles_b[0][self.set_order_b[recognized_colors[i]]] = self.filtered_circles2[0][i]
            elif recognized_colors[i] == reordered_list_b[1]:
                reordered_circles_b[0][self.set_order_b[recognized_colors[i]]] = self.filtered_circles2[0][i]
            elif recognized_colors[i] == reordered_list_b[2]:
                reordered_circles_b[0][self.set_order_b[recognized_colors[i]]] = self.filtered_circles2[0][i]
            elif recognized_colors[i] == reordered_list_b[3]:
                reordered_circles_b[0][self.set_order_b[recognized_colors[i]]] = self.filtered_circles2[0][i]
            else:
                continue
                # print('color not found/not in set')
                
        return reordered_circles_f, reordered_circles_b
    
    def vectorize(self):
        reordered_circles_f, reordered_circles_b = CircleDetect.reorder_circles(self)
        
        vectors_f = np.zeros((3,2), dtype=np.int32)
        vectors_b = np.zeros((3,2), dtype=np.int32)
        
        for i in range(len(reordered_circles_f[0])-1):
            if i == 0:
                vectors_f[0][0] = reordered_circles_f[0][i+1][0] - reordered_circles_f[0][i][0]
                vectors_f[0][1] = reordered_circles_f[0][i+1][1] - reordered_circles_f[0][i][1]
            elif i == 1:
                vectors_f[1][0] = reordered_circles_f[0][i+1][0] - reordered_circles_f[0][i][0]
                vectors_f[1][1] = reordered_circles_f[0][i+1][1] - reordered_circles_f[0][i][1]
            elif i == 2:
                vectors_f[2][0] = reordered_circles_f[0][i+1][0] - reordered_circles_f[0][i][0]
                vectors_f[2][1] = reordered_circles_f[0][i+1][1] - reordered_circles_f[0][i][1]
        
        for i in range(len(reordered_circles_b[0])-1):
            if i == 0:
                vectors_b[0][0] = reordered_circles_b[0][i+1][0] - reordered_circles_b[0][i][0]
                vectors_b[0][1] = reordered_circles_b[0][i+1][1] - reordered_circles_b[0][i][1]
            elif i == 1:
                vectors_b[1][0] = reordered_circles_b[0][i+1][0] - reordered_circles_b[0][i][0]
                vectors_b[1][1] = reordered_circles_b[0][i+1][1] - reordered_circles_b[0][i][1]
            elif i == 2:
                vectors_b[2][0] = reordered_circles_b[0][i+1][0] - reordered_circles_b[0][i][0]
                vectors_b[2][1] = reordered_circles_b[0][i+1][1] - reordered_circles_b[0][i][1]
                
        return vectors_f, vectors_b
        
    
    def process_image(self):
        reordered_circles_f, reordered_circles_b = CircleDetect.reorder_circles(self)
                
        cimg_copy = cv2.cvtColor(self.cimg.copy(), cv2.COLOR_RGB2BGR)
        vectors_f = np.zeros((3,2), dtype=np.int32)
        vectors_b = np.zeros((3,2), dtype=np.int32)
        
        for i in range(len(reordered_circles_f[0])-1):
            # draw line
            if i == 0:  
                cv2.line(img=cimg_copy, pt1=(reordered_circles_f[0][i][0], reordered_circles_f[0][i][1]), pt2=(reordered_circles_f[0][i+1][0], reordered_circles_f[0][i+1][1]), color=(255,0,0), thickness=5)
                vectors_f[0][0] = reordered_circles_f[0][i+1][0] - reordered_circles_f[0][i][0]
                vectors_f[0][1] = reordered_circles_f[0][i+1][1] - reordered_circles_f[0][i][1]
            elif i == 1:
                cv2.line(img=cimg_copy, pt1=(reordered_circles_f[0][i][0], reordered_circles_f[0][i][1]), pt2=(reordered_circles_f[0][i+1][0], reordered_circles_f[0][i+1][1]), color=(0,255,0), thickness=5)
                vectors_f[1][0] = reordered_circles_f[0][i+1][0] - reordered_circles_f[0][i][0]
                vectors_f[1][1] = reordered_circles_f[0][i+1][1] - reordered_circles_f[0][i][1]
            elif i == 2:
                cv2.line(img=cimg_copy, pt1=(reordered_circles_f[0][i][0], reordered_circles_f[0][i][1]), pt2=(reordered_circles_f[0][i+1][0], reordered_circles_f[0][i+1][1]), color=(0,0,255), thickness=5)
                vectors_f[2][0] = reordered_circles_f[0][i+1][0] - reordered_circles_f[0][i][0]
                vectors_f[2][1] = reordered_circles_f[0][i+1][1] - reordered_circles_f[0][i][1]
                
        for i in range(len(reordered_circles_b[0])-1):
            # draw line
            if i == 0:  
                cv2.line(img=cimg_copy, pt1=(reordered_circles_b[0][i][0], reordered_circles_b[0][i][1]), pt2=(reordered_circles_b[0][i+1][0], reordered_circles_b[0][i+1][1]), color=(255,0,0), thickness=5)
                vectors_b[0][0] = reordered_circles_b[0][i+1][0] - reordered_circles_b[0][i][0]
                vectors_b[0][1] = reordered_circles_b[0][i+1][1] - reordered_circles_b[0][i][1]
            elif i == 1:
                cv2.line(img=cimg_copy, pt1=(reordered_circles_b[0][i][0], reordered_circles_b[0][i][1]), pt2=(reordered_circles_b[0][i+1][0], reordered_circles_b[0][i+1][1]), color=(0,255,0), thickness=5)
                vectors_b[1][0] = reordered_circles_b[0][i+1][0] - reordered_circles_b[0][i][0]
                vectors_b[1][1] = reordered_circles_b[0][i+1][1] - reordered_circles_b[0][i][1]
            elif i == 2:
                cv2.line(img=cimg_copy, pt1=(reordered_circles_b[0][i][0], reordered_circles_b[0][i][1]), pt2=(reordered_circles_b[0][i+1][0], reordered_circles_b[0][i+1][1]), color=(0,0,255), thickness=5)
                vectors_b[2][0] = reordered_circles_b[0][i+1][0] - reordered_circles_b[0][i][0]
                vectors_b[2][1] = reordered_circles_b[0][i+1][1] - reordered_circles_b[0][i][1]
                
        return vectors_f, vectors_b, cimg_copy
    
    def show_processed_image(self):
        vectors_f, vectors_b, cimg_copy = CircleDetect.process_image(self)

        for i in self.filtered_circles2[0,:]:
            # draw the outer circle
            cv2.circle(cimg_copy,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg_copy,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow('detected circles', cimg_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return vectors_f, vectors_b
        
    def show_raw_circles(self, isgray=False):
        if isgray:
            rawimg = self.img.copy()
        else:
            rawimg = self.cimg.copy()
            rawimg = cv2.cvtColor(rawimg, cv2.COLOR_RGB2BGR)
        
        for i in self.circles[0,:]:
            # draw the outer circle
            cv2.circle(rawimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(rawimg,(i[0],i[1]),2,(0,0,255),3)
            
        cv2.imshow('initial detected circles',rawimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_initial_circles(self, isgray=False):
        if isgray:
            icimg = self.img.copy()
        else:
            icimg = self.cimg.copy()
            icimg = cv2.cvtColor(icimg, cv2.COLOR_RGB2BGR)

        for i in self.filtered_circles[0,:]:
            # draw the outer circle
            cv2.circle(icimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(icimg,(i[0],i[1]),2,(0,0,255),3)
        # cv2.namedWindow('detected circles',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('detected circles', 1800, int((2256/4000)*1800))
        cv2.imshow('initial detected circles',icimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def show_filtered_circles(self, isgray=False):
        
        filtered_circles, recognized_colors = CircleDetect.anti_duplicates(self)
        self.filtered_circles2 = filtered_circles
        print('\n')
        print(recognized_colors)
        
        if isgray:
            fcimg = self.img.copy()
        else:
            fcimg = self.cimg.copy()

        for i in self.filtered_circles2[0,:]:
            # draw the outer circle
            cv2.circle(fcimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(fcimg,(i[0],i[1]),2,(0,0,255),3)
        # cv2.namedWindow('detected circles',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('detected circles', 1800, int((2256/4000)*1800))
        cv2.imshow('filtered detected circles',fcimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def threshold_img(self):
        cimg = self.cimg.copy()
        cimg_thresh = np.zeros_like(cimg)
        cimg_thresh[:,:,0] = cv2.adaptiveThreshold(cimg[:,:,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 12)
        cimg_thresh[:,:,1] = cv2.adaptiveThreshold(cimg[:,:,1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 12)
        cimg_thresh[:,:,2] = cv2.adaptiveThreshold(cimg[:,:,2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 12) 
        
        circlesr = cv2.HoughCircles(cimg_thresh[:,:,0],cv2.HOUGH_GRADIENT,1,self.minimumDist,
                            param1=self.parameter1,param2=self.parameter2,minRadius=self.minimumRadius,maxRadius=self.maximumRadius)
        circlesr = np.int32(np.around(circlesr))
        
        circlesg = cv2.HoughCircles(cimg_thresh[:,:,1],cv2.HOUGH_GRADIENT,1,self.minimumDist,
                            param1=self.parameter1,param2=self.parameter2,minRadius=self.minimumRadius,maxRadius=self.maximumRadius)
        circlesg = np.int32(np.around(circlesg))
        
        circlesb = cv2.HoughCircles(cimg_thresh[:,:,2],cv2.HOUGH_GRADIENT,1,self.minimumDist,
                            param1=self.parameter1,param2=self.parameter2,minRadius=self.minimumRadius,maxRadius=self.maximumRadius)
        circlesb = np.int32(np.around(circlesb))
        
        combined_circles = np.concatenate((circlesr, circlesg, circlesb), axis=1)
        
        # colored_img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
        
        for i in combined_circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg_thresh,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg_thresh,(i[0],i[1]),2,(0,0,255),3)
        
        cv2.imshow('post threshold',cimg_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def threshold_img_otsu(self):
        cimg = self.cimg.copy()
        cimg_thresh = np.zeros_like(cimg)
        cimgr_thresh = np.array(cimg[:,:,0])
        cimgg_thresh = np.array(cimg[:,:,1])
        cimgb_thresh = np.array(cimg[:,:,2])
        
        ret1, cimgr_thresh = cv2.threshold(cimgr_thresh, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret2, cimgg_thresh = cv2.threshold(cimgg_thresh, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret3, cimgb_thresh = cv2.threshold(cimgb_thresh, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
           
        # ret1, cimg_thresh[:,:,0] = cv2.threshold(cimg[:,:,0], 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret2, cimg_thresh[:,:,1] = cv2.threshold(cimg[:,:,1], 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret3, cimg_thresh[:,:,2] = cv2.threshold(cimg[:,:,2], 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        cimg_thresh[:,:,0] = cimgr_thresh
        cimg_thresh[:,:,1] = cimgg_thresh
        cimg_thresh[:,:,2] = cimgb_thresh
        
        circlesr = cv2.HoughCircles(cimgr_thresh,cv2.HOUGH_GRADIENT,1,self.minimumDist,
                            param1=self.parameter1,param2=self.parameter2,minRadius=self.minimumRadius,maxRadius=self.maximumRadius)
        
        circlesg = cv2.HoughCircles(cimgg_thresh,cv2.HOUGH_GRADIENT,1,self.minimumDist,
                            param1=self.parameter1,param2=self.parameter2,minRadius=self.minimumRadius,maxRadius=self.maximumRadius)
        
        circlesb = cv2.HoughCircles(cimgb_thresh,cv2.HOUGH_GRADIENT,1,self.minimumDist,
                            param1=self.parameter1,param2=self.parameter2,minRadius=self.minimumRadius,maxRadius=self.maximumRadius)
        
        print(type(circlesr))
        print(type(circlesg))
        print(type(circlesb))
        
        # print(circlesr.shape)
        # print(circlesg.shape)
        # print(circlesb.shape)   
        
        if isinstance(circlesr, np.ndarray):
            pass
        else:
            circlesr = np.array([[[0,0,0]]])
        
        if isinstance(circlesg, np.ndarray):
            pass
        else:
            circlesg = np.array([[[0,0,0]]])
            
        if isinstance(circlesb, np.ndarray):
            pass
        else:
            circlesb = np.array([[[0,0,0]]])     
        
        circlesr = np.int32(np.around(circlesr))
        circlesg = np.int32(np.around(circlesg))
        circlesb = np.int32(np.around(circlesb))
        
        combined_circles = np.concatenate((circlesr, circlesg, circlesb), axis=1)
        
        # colored_img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
        
        for i in combined_circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg_thresh,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg_thresh,(i[0],i[1]),2,(0,0,255),3)
        
        cv2.imshow('post threshold',cimg_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        