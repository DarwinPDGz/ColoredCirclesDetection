import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import PIL
# import pandas as pd
# import openpyxl as op

class CircleDetectInstance:
    
    def __init__(self, image_path, cct=0, range_type=0, arrayVersion=False, thresholdVersion=False, detectFULL=False, altFilter=True, minimumDist=50, parameter1=70, parameter2=30, minimumRadius=15, maximumRadius=30, tolerance=5, ptpthreshold=15, set_order_f=None, set_order_b=None, autoparam=False):
        self.image_path = image_path
        self.cct = cct
        self.range_type = range_type
        self.arrayVersion = arrayVersion
        self.thresholdVersion = thresholdVersion
        self.detectFULL = detectFULL
        self.altFilter = altFilter
        self.minimumDist = minimumDist
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.minimumRadius = minimumRadius
        self.maximumRadius = maximumRadius
        self.tolerance = tolerance
        self.ptpthreshold = ptpthreshold
        self.set_order_f = set_order_f
        self.set_order_b = set_order_b
        
        self.exceptions = True
        self.re_parameter2 = parameter2
        self.cd = None
        
        if autoparam:
            readjust = True
            self.re_parameter2 = 40
            # self.re_minimumRadius = None
            # self.re_maximumRadius = None
            
            while readjust:
                try:
                    benchmark_instance = CircleDetect(self.image_path, self.cct, self.range_type, self.arrayVersion, self.thresholdVersion, self.detectFULL, self.altFilter, self.minimumDist, parameter1=70, parameter2=self.re_parameter2, minimumRadius=self.minimumRadius, maximumRadius=self.maximumRadius, tolerance=self.tolerance, ptpthreshold=self.ptpthreshold, set_order_f=self.set_order_f, set_order_b=self.set_order_b, benchmark=True)

                except AttributeError:
                    self.re_parameter2 -= 1
                    print('re_parameter2 (NOT done, setting radius): ', self.re_parameter2)
                    
                # except Exception as e:
                #     print(e, 'readjusting, radius')
                #     print(self.image_path)
                #     self.exceptions = False
                #     # return 0, 0
                
                else:
                    self.minimumRadius = benchmark_instance.minmode_Radius
                    self.maximumRadius = benchmark_instance.maxmode_Radius
                    
                    print('re_parameter2 (done, setting radius): ', self.re_parameter2)
                    print('re_minimumRadius: ', self.minimumRadius)
                    print('re_maximumRadius: ', self.maximumRadius)
                    readjust = False
                    break
                
            while self.exceptions:
                if self.re_parameter2 > 10:
                    try:
                        self.cd = CircleDetect(self.image_path, self.cct, self.range_type, self.arrayVersion, self.thresholdVersion, self.detectFULL, self.altFilter, self.minimumDist, self.parameter1, self.re_parameter2, self.minimumRadius, self.maximumRadius, self.tolerance, self.ptpthreshold, self.set_order_f, self.set_order_b)
                    except (AttributeError, ValueError):
                        self.exceptions = True
                        self.re_parameter2 -= 1
                        print('re_parameter2 (NOT done, instance is autoparamed): ', self.re_parameter2)
                    # except Exception as e:
                    #     print(e, 'readjusting, param2')
                    #     print(self.image_path)
                    #     self.exceptions = True
                    #     # return 0, 0
                    else:
                        print('re_parameter2 (done, instance is autoparamed): ', self.re_parameter2)
                        self.exceptions = False
                        break
                else:
                    print("bad image, skipping")
                    break
                
        else:
            # run an instance normally
            # self.cd = CircleDetect(self.image_path, self.cct, self.range_type, self.arrayVersion, self.thresholdVersion, self.detectFULL, self.altFilter, self.minimumDist, self.parameter1, self.parameter2, self.minimumRadius, self.maximumRadius, self.tolerance, self.ptpthreshold, self.set_order_f, self.set_order_b)
            
            while self.exceptions:
                if self.re_parameter2 > 10:
                    try:
                        self.cd = CircleDetect(self.image_path, self.cct, self.range_type, self.arrayVersion, self.thresholdVersion, self.detectFULL, self.altFilter, self.minimumDist, self.parameter1, self.parameter2, self.minimumRadius, self.maximumRadius, self.tolerance, self.ptpthreshold, self.set_order_f, self.set_order_b)
                    except ValueError:
                        self.exceptions = True
                        self.parameter2 -= 1
                        print('re_parameter2: (NOT done, instance done normally)', self.parameter2)
                    # except Exception as e:
                    #     print(e, 'readjusting, param2, no autoparam')
                    #     print(self.image_path)
                    #     self.exceptions = True
                    #     # return 0, 0
                    else:
                        print('re_parameter2 (done, instance done normally): ', self.re_parameter2)
                        self.exceptions = False
                        break
                else:
                    print("bad image, skipping")
                    break

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
        'cyan': 3,
        'magenta': 2,
        'yellow': 1}
    
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
      
    def __init__(self, image_path, cct=0, range_type=0, arrayVersion=False, thresholdVersion=False, detectFULL=False, altFilter=True, minimumDist=20, parameter1=70, parameter2=20, minimumRadius=0, maximumRadius=30, tolerance=5, ptpthreshold=15, set_order_f=None, set_order_b=None, benchmark=False):
        if set_order_f or set_order_f is None:
            RuntimeError('set_order_f/set_order_b must be set')
        
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
        self.altFilter = altFilter
        self.minmode_Radius = None
        self.maxmode_Radius = None
        self.htlf_dict = {}
        
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
            
        if benchmark:
            mode = self.benchmark_radius()
            minmode = mode - 4
            maxmode = mode + 4
            
            print('minmode: ', minmode)
            print('maxmode: ', maxmode)
            
            self.minmode_Radius = minmode
            self.maxmode_Radius = maxmode
            
        else:
            # debug functions
            # self.show_raw_circles()
            
            if self.altFilter:
                self.filtered_circles = CircleDetect.alt_filter_circles(self)
                self.filtered_circles2 = self.filtered_circles
                
            else:
                self.filtered_circles = CircleDetect.filter_circles(self)
                self.filtered_circles2 = self.filtered_circles
                
            print(self.filtered_circles.shape)
            
            self.vf, self.vb, unused_cimg = CircleDetect.process_image(self) 
        
    def benchmark_radius(self):
        circles = self.circles
        
        radius_list = []
        radius_mode = []
        radius_counter = []
        
        
        for i in range(len(circles[0])):
            x = circles[0][i][0]
            y = circles[0][i][1]
            r = circles[0][i][2]
            
            radius_list.append(r)
            
            # print('radius: ', r)
            # self.get_range(x, y, r, self.cimg)
        
        radius_list = np.array(radius_list)
        
        radius_list = np.sort(radius_list)
        
        tracker = 0
        
        for i in range(len(radius_list)):
            if i == 0:
                radius_mode.append(radius_list[i])
                radius_counter.append(1)
            else:
                if radius_list[i] == radius_list[i-1]:
                    radius_counter[tracker] += 1
                else:
                    radius_mode.append(radius_list[i])
                    radius_counter.append(1)
                    tracker += 1
        
        print('radius list', radius_list)
        
        print('radius mode', radius_mode)
        
        print('radius counter', radius_counter)
        
        radius_mode = np.array(radius_mode)
        radius_counter = np.array(radius_counter)
        
        highest_radius_counter = np.argsort(radius_counter)[::-1][:2]
        which_radius_mode = radius_mode[highest_radius_counter]
        
        print('which radius mode', which_radius_mode[0], which_radius_mode[1])
        
        return int((which_radius_mode[0] + which_radius_mode[1])/2)
        
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
                    color_array[0][i], color_array[1][i], color_array[2][i] = 150, 68, 59 #v
                elif color == 'green':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 84, 110, 45 #v
                elif color == 'blue':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 51, 62, 126 #v
                elif color == 'cyan':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 103, 137, 174 #v
                elif color == 'magenta':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 151, 73, 105 #v
                elif color == 'yellow':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 201, 181, 20 #v
                elif color == 'black':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 37, 36, 32 #v
                else:
                    continue
                    # print('color not found')
        else:
            for i, color in enumerate(CircleDetect.COLOR_LIST):
                if color == 'red':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 147, 60, 54
                elif color == 'green':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 76, 96, 57
                elif color == 'blue':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 52, 61, 120
                elif color == 'cyan':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 80, 101, 132
                elif color == 'magenta':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 137, 62, 93
                elif color == 'yellow':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 163, 144, 23
                elif color == 'black':
                    color_array[0][i], color_array[1][i], color_array[2][i] = 27, 26, 22
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
    
    def lex_sorter(self, circleposx, circleposy, radius, img, sorted=True):
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
        
        if sorted:
            sorted_h = np.sort(new_list_h)
            sorted_l = np.sort(new_list_l)
            sorted_s = np.sort(new_list_s)
            
            # np.set_printoptions(threshold=np.inf)
            # print(sorted_h)
            # print('\n')
            # print(sorted_l)
            # print('\n')
            # print(sorted_s)
            # np.set_printoptions(threshold=1000)
            
            return sorted_h, sorted_l, sorted_s
        
        else:
            return new_list_h, new_list_l, new_list_s
        
        
        
    
    def grab_big_groups(self, circleposx, circleposy, radius, img):
        h, l, s = CircleDetect.lex_sorter(self, circleposx, circleposy, radius, img)
        
        hues_to_look_for = []
        
        h_group = []
        h_counter = []
        
        for i in range(len(h)):
            if i == 0:
                h_group.append(h[i])
                h_counter.append(1)
            else:
                if h[i] == h[i-1]:
                    h_counter[-1] += 1
                else:
                    h_group.append(h[i])
                    h_counter.append(1)
        # print(h_group)
        # print(h_counter)   
                 
        h_group = np.array(h_group)
        h_counter = np.array(h_counter)
        
        highest_h_counter = np.argsort(h_counter)[::-1][:(len(h_counter))]
        which_h_group = h_group[highest_h_counter[:(len(h_counter))]]
        h_counter_amount = h_counter[highest_h_counter[:(len(h_counter))]]                    
        
        # print(highest_h_counter)
        # print(which_h_group)
        
        l_group = []
        l_counter = []
        
        for i in range(len(l)):
            if i == 0:
                l_group.append(l[i])
                l_counter.append(1)
            else:
                if l[i] == l[i-1]:
                    l_counter[-1] += 1
                else:
                    l_group.append(l[i])
                    l_counter.append(1)
                    
        l_group = np.array(l_group)
        l_counter = np.array(l_counter)
              
        highest_l_counter = np.argsort(l_counter)[::-1][:(len(l_counter))]
        which_l_group = l_group[highest_l_counter[:(len(l_counter))]]
        l_counter_amount = l_counter[highest_l_counter[:(len(l_counter))]]
        
        # print(l_group)
        # print(l_counter)                
        # print(highest_l_counter)
        # print(which_l_group)
        
        s_group = []
        s_counter = []
        
        for i in range(len(s)):
            if i == 0:
                s_group.append(s[i])
                s_counter.append(1)
            else:
                if s[i] == s[i-1]:
                    s_counter[-1] += 1
                else:
                    s_group.append(s[i])
                    s_counter.append(1)
                    
        s_group = np.array(s_group)
        s_counter = np.array(s_counter)
             
        highest_s_counter = np.argsort(s_counter)[::-1][:(len(s_counter))]
        which_s_group = s_group[highest_s_counter[:(len(s_counter))]]
        s_counter_amount = s_counter[highest_s_counter[:(len(s_counter))]]
        
        # print(s_group)
        # print(s_counter)  
        # print(highest_s_counter)
        # print(which_s_group)
                 
        h_cumulative = 0
        l_cumulative = 0
        s_cumulative = 0
        
        h_check = False
        l_check = False
        s_check = False
        
        # lightness check
        for i in range(len(which_l_group)):
            if i == 0:
                l_cumulative += l_counter_amount[i]
                l_baseline = which_l_group[i]
                
            else:
                if abs(int(which_l_group[i]) - int(l_baseline)) < 10:
                    l_cumulative += l_counter_amount[i]
                else:
                    continue
                
        if l_cumulative > int(len(l)*0.7):
            l_check = True
            
        # hue check
        for i in range(len(which_h_group)):
            if i == 0:
                h_cumulative += h_counter_amount[i]
                h_baseline = which_h_group[i]
                hues_to_look_for.append(h_baseline)
                
            else:
                if abs(int(which_h_group[i]) - int(h_baseline)) < 10:
                    h_cumulative += h_counter_amount[i]
                    hues_to_look_for.append(which_h_group[i])
                else:
                    continue
                
        if h_cumulative > int(len(h)*0.5):
            h_check = True
            
        # saturation check
        for i in range(len(which_s_group)):
            if i == 0:
                s_cumulative += s_counter_amount[i]
                s_baseline = which_s_group[i]
                
            else:
                if abs(int(which_s_group[i]) -int(s_baseline)) < 10:
                    s_cumulative += s_counter_amount[i]
                else:
                    continue
        
        if s_cumulative > int(len(s)*0.2):
            s_check = True
            
        ratio_h = h_cumulative/len(h)
        ratio_l = l_cumulative/len(l)
        ratio_s = s_cumulative/len(s)
        
        print('ratio')
        
        print('h: ', ratio_h)
        print('l: ', ratio_l)
        print('s: ', ratio_s)
        
        hues_to_look_for = np.array(hues_to_look_for)
        
        print(hues_to_look_for, hues_to_look_for.shape)
            
        return h_check, l_check, s_check, hues_to_look_for
    
    def view_roc(self, circleposx, circleposy, radius, img):
        h, l, s = CircleDetect.lex_sorter(self, circleposx, circleposy, radius, img)
        
        h_roc = np.zeros_like(h, dtype=np.uint8)
        l_roc = np.zeros_like(l, dtype=np.uint8)
        s_roc = np.zeros_like(s, dtype=np.uint8)
        
        print('view roc')
        print(h_roc.shape)
        print(l_roc.shape)
        print(s_roc.shape)
        
        for i in range(len(h)):
            if i == 0:
                h_roc[i] = 0
                l_roc[i] = 0
                s_roc[i] = 0
            else:
                h_roc[i] = h[i] - h[i-1]
                l_roc[i] = l[i] - l[i-1]
                s_roc[i] = s[i] - s[i-1]
                
        fig = plt.figure(figsize=(20,30))
        fig.add_subplot(3,1,1)
        plt.plot(np.array(range(len(h_roc))), h_roc, color='red')
        fig.add_subplot(3,1,2)
        plt.plot(np.array(range(len(l_roc))), l_roc, color='green')
        fig.add_subplot(3,1,3)
        plt.plot(np.array(range(len(s_roc))), s_roc, color='blue')
        
        
    
    def hls_visualizer(self, circleposx, circleposy, radius, img):
        h, l, s = CircleDetect.lex_sorter(self, circleposx, circleposy, radius, img)
        
        h_strip = np.zeros((20,len(h),3), dtype=np.uint8)
        l_strip = np.zeros((20,len(l),3), dtype=np.uint8)
        s_strip = np.zeros((20,len(s),3), dtype=np.uint8)
        
        print(h_strip.shape)
        print(l_strip.shape)
        print(s_strip.shape)
    
        for i in range(len(h)):
            # print(i)
            # convert hue into color wheel/strip
            h_strip[0,i] = [h[i], 127, 255]
            l_strip[0,i] = [0, l[i], 255]
            s_strip[0,i] = [0, 127, s[i]]
            
        for i in range(1,20):
            h_strip[i] = h_strip[0]
            l_strip[i] = l_strip[0]
            s_strip[i] = s_strip[0]
            
        h_strip = cv2.cvtColor(h_strip, cv2.COLOR_HLS2RGB)
        l_strip = cv2.cvtColor(l_strip, cv2.COLOR_HLS2RGB)
        s_strip = cv2.cvtColor(s_strip, cv2.COLOR_HLS2RGB)
        
        fig = plt.figure(figsize=(15,7))
        fig.add_subplot(3,1,1)
        plt.imshow(h_strip)
        fig.add_subplot(3,1,2)
        plt.imshow(l_strip)
        fig.add_subplot(3,1,3)
        plt.imshow(s_strip)
        
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
    
    def dE94_comparator(self, circleposx, circleposy, radius, cimg, alt=False, n=0):
        color_array = self.color_array
    
        list_of_colors = np.zeros((1,7,3), dtype=np.uint8)
        
        mean_L = None
        mean_A = None
        mean_B = None
        
        lab_cimg = None
        
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
        
        if alt:
            hls = CircleDetect.process_htlf(self, circleposx, circleposy, radius, n)
            
            hls = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
            
            lab_cimg = cv2.cvtColor(hls, cv2.COLOR_RGB2LAB)
            
            mean_L = np.mean(lab_cimg[:,:,0])
            mean_A = np.mean(lab_cimg[:,:,1])
            mean_B = np.mean(lab_cimg[:,:,2])
        
        else:
        
            lab_cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2LAB)
            
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
        
        ##########
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
            # if delta_E_array[0] > 17:
            #     determined_color = 'inconclusive'
            # else:
            #     determined_color = 'red'
            # print('red')
            determined_color = 'red'
        elif delta_E_array.min() == delta_E_array[1]:
            # if delta_E_array[1] > 11:
            #     determined_color = 'inconclusive'
            # else:
            #     determined_color = 'green'
            determined_color = 'green'
            # print('green')
        elif delta_E_array.min() == delta_E_array[2]:
            # if delta_E_array[2] > 12:
            #     determined_color = 'inconclusive'
            # else:
            #     determined_color = 'blue'
            determined_color = 'blue'
            # print('blue')
        elif delta_E_array.min() == delta_E_array[3]:
            # if delta_E_array[3] > 12:
            #     determined_color = 'inconclusive'
            # else:
            #     determined_color = 'cyan'
            determined_color = 'cyan'
            # print('cyan')
        elif delta_E_array.min() == delta_E_array[4]:
            # if delta_E_array[4] > 11:
            #     determined_color = 'inconclusive'
            # else:
            #     determined_color = 'magenta'
            determined_color = 'magenta'
            # print('magenta')
        elif delta_E_array.min() == delta_E_array[5]:
            # if delta_E_array[5] > 11:
            #     determined_color = 'inconclusive'
            # else:
            #     determined_color = 'yellow'
            determined_color = 'yellow'
            # print('yellow')
        elif delta_E_array.min() == delta_E_array[6]:
            # if delta_E_array[6] > 11:
            #     print('inconclusive black: ' + str(delta_E_array[6]))
            #     determined_color = 'inconclusive'
            # else:
            #     determined_color = 'black'
                
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
        # hue, lightness, saturation check
        circles = self.circles
        
        delete_list = []
        dictCounter = 0
                    
        for i in range(circles.shape[1]):
            h_check, l_check, s_check, htlf = CircleDetect.grab_big_groups(self, circles[0][i][0], circles[0][i][1], circles[0][i][2], self.img)
            
            trues = 0

            
            # if 2/3 checks are true, keep
            if h_check == True:
                trues += 1
            if l_check == True:
                trues += 1
            if s_check == True:
                trues += 1
                
            if trues < 2:
                delete_list.append(i)
            else:
                print('alt_filter_circles passed:' , i)
                ### cursed, make dynamic dictionary
                self.htlf_dict[dictCounter] = np.array(htlf)
                dictCounter += 1
                
                
        print('htlf dict', self.htlf_dict.keys())
                
        delete_list = sorted(delete_list, reverse=True)
        
        for i in delete_list:
            circles = np.delete(circles, i, axis=1)
            
        filtered_circles = circles
        
        return filtered_circles
    
    def process_htlf(self, circleposx, circleposy, radius, n):
        
        h, l, s = CircleDetect.lex_sorter(self, circleposx, circleposy, radius, self.img, sorted=False)
        
        htlf_array = self.htlf_dict[n]
        
        h_list = []
        l_list = []
        s_list = []
        
        for j in range(len(h)):
            if h[j] in htlf_array:
                h_list.append(h[j])
                l_list.append(l[j])
                s_list.append(s[j])
            else:
                continue
        
        h_list = np.array(h_list)
        l_list = np.array(l_list)
        s_list = np.array(s_list)
        
        reconstructed_hls = np.zeros((1, h_list.shape[0], 3), dtype=np.uint8)
        
        reconstructed_hls[:,:,0] = h_list
        reconstructed_hls[:,:,1] = l_list
        reconstructed_hls[:,:,2] = s_list
        
        return reconstructed_hls
        
     
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
        
        if self.altFilter:
            for i in range(filtered_circles.shape[1]):
                rcol, ifac = CircleDetect.dE94_comparator(self, filtered_circles[0][i][0], filtered_circles[0][i][1], filtered_circles[0][i][2], self.cimg, alt=True, n=i)
                recognized_colors.append(rcol)
                instability_factor.append(ifac)           
            
        else:
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
        
        # print(filtered_circles)
        # print(filtered_circles.shape)
        
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
            
        for i in range(7):
            if i==0:
                if len(red_list) == 0:
                    raise ValueError('No red detected')
            elif i==1:
                if len(green_list) == 0:
                    raise ValueError('No green detected')
            elif i==2:
                if len(blue_list) == 0:
                    raise ValueError('No blue detected')
            elif i==3:
                if len(cyan_list) == 0:
                    raise ValueError('No cyan detected')
            elif i==4:
                if len(magenta_list) == 0:
                    raise ValueError('No magenta detected')
            elif i==5:
                if len(yellow_list) == 0:
                    raise ValueError('No yellow detected')
            elif i==6:
                if len(black_list) == 0:
                    raise ValueError('No black detected')
            else:
                continue
                
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
        
    
    def process_image(self, save=False, path='processed_image.jpg'):
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
                
        if save:
            cv2.imwrite(path, cimg_copy)
                
        return vectors_f, vectors_b, cimg_copy
    
    def show_processed_image(self, save=False, path='processed_image.jpg'):
        vectors_f, vectors_b, cimg_copy = CircleDetect.process_image(self)

        for i in self.filtered_circles2[0,:]:
            # draw the outer circle
            cv2.circle(cimg_copy,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg_copy,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow('detected circles', cimg_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if save:
            cv2.imwrite(path, cimg_copy)
        
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
            fcimg = cv2.cvtColor(fcimg, cv2.COLOR_RGB2BGR)

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
        
    def show_singular_circle(self, circleposx, circleposy, radius):
        cimg = self.cimg.copy()
        cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR)
        cv2.circle(cimg,(circleposx,circleposy),radius,(0,255,0),2)
        cv2.circle(cimg,(circleposx,circleposy),2,(0,0,255),3)
        cv2.imshow('singular circle',cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()