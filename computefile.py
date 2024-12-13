from circledetectexperimental import CircleDetect, CircleDetectInstance
import os

class ComputeFile:
    
    def __init__(self, path, issita=False, israma=False, issugriwa=False, issubali=False, israwanamf=False, ishanuman=False, islaksmana=False, iswibhisana=False, ishanumanalt=False, issubalialt=False, ishanumanpts=False, israwanapts=False, isangada=False, isanila=False, isangada2=False, isanila2=False):
        """processes all the images in the path specified using the functions from circledetectexperimental.py, creates nested dictionaries for each instance per image in the file"""
        self.path = []
        self.path.append(path)

        # path = []

        # path.append('c:/Users/Darwin/Documents/Wayang_related/imgprocessing/sita_initial_wflash/threequarters/pts/')
        # path.append('c:/Users/Darwin/Documents/Wayang_related/imgprocessing/sita_initial_wflash/threequarters/lptf/')
        # path.append('c:/Users/Darwin/Documents/Wayang_related/imgprocessing/rahwana/ptf/')
        # path.append('c:/Users/Darwin/Documents/Wayang_related/imgprocessing/rahwana/lptf/')
        # path.append('c:/Users/Darwin/Documents/Wayang_related/imgprocessing/rahwana/df/')
        # path.append('c:/Users/Darwin/Documents/Wayang_related/imgprocessing/rahwana/pts/') no
        # path.append('c:/Users/Darwin/Documents/Wayang_related/imgprocessing/rahwana/mf/')

        print(self.path)

        dict_of_dirs = {}

        for i in range(len(self.path)):
            list = os.listdir(self.path[i])
            dict_of_dirs[i] = list
            
        dict_of_real_paths = {}

        for i in range(len(self.path)):
            real_path = []
            for j in range(len(dict_of_dirs[i])):
            # for j in range(1):
                real_path.append(self.path[i] + dict_of_dirs[i][j])
            dict_of_real_paths[i] = real_path

        dict_list_of_params = {}

        dict_list_of_params_folder = {}

        for i in range(len(dict_of_real_paths)):
            for j in range(len(dict_of_real_paths[i])):
                if issita:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 20,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_SITA,
                        'set_order_b':CircleDetect.SET_ORDER_B_SITA,
                        'arrayVersion': True}
                else:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 15,
                        'minRadius': 19,
                        'maxRadius': 29,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_RAWANA_PTS,
                        'set_order_b':CircleDetect.SET_ORDER_B_RAWANA_PTS,
                        'arrayVersion': False}
                    
                if israma:
                    list_of_params = {}
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 30,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_RAMA,
                        'set_order_b':CircleDetect.SET_ORDER_B_RAMA,
                        'arrayVersion': True}
                    
                if issugriwa:
                    list_of_params = {}
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_SUGRIWA,
                        'set_order_b':CircleDetect.SET_ORDER_B_SUGRIWA,
                        'arrayVersion': 4}
                    
                if issubali:
                    list_of_params = {}
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_BALI,
                        'set_order_b':CircleDetect.SET_ORDER_B_BALI,
                        'arrayVersion': 5}
                    
                if israwanamf:
                    list_of_params = {}
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_RAWANA,
                        'set_order_b':CircleDetect.SET_ORDER_B_RAWANA,
                        'arrayVersion': 11}
                    
                if ishanuman:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 50,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_HANUMAN,
                        'set_order_b':CircleDetect.SET_ORDER_B_HANUMAN,
                        'arrayVersion': 2}
                    
                if islaksmana:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_LAKSMANA,
                        'set_order_b':CircleDetect.SET_ORDER_B_LAKSMANA,
                        'arrayVersion': 3}
                    
                if iswibhisana:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_WIBHISANA,
                        'set_order_b':CircleDetect.SET_ORDER_B_WIBHISANA,
                        'arrayVersion': 6}
                    
                if ishanumanalt:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_HANUMAN,
                        'set_order_b':CircleDetect.SET_ORDER_B_HANUMAN,
                        'arrayVersion': 12}
                    
                if issubalialt:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_BALI,
                        'set_order_b':CircleDetect.SET_ORDER_B_BALI,
                        'arrayVersion': 13}
                    
                if ishanumanpts:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_HANUMAN,
                        'set_order_b':CircleDetect.SET_ORDER_B_HANUMAN,
                        'arrayVersion': 14}
                    
                if israwanapts:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_RAWANA_PTS,
                        'set_order_b':CircleDetect.SET_ORDER_B_RAWANA_PTS,
                        'arrayVersion': 15}
                    
                if isangada:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_ANGADA,
                        'set_order_b':CircleDetect.SET_ORDER_B_ANGADA,
                        'arrayVersion': 17}
                    
                if isanila:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_ANILA,
                        'set_order_b':CircleDetect.SET_ORDER_B_ANILA,
                        'arrayVersion': 18}

                if isangada2:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_ANGADA,
                        'set_order_b':CircleDetect.SET_ORDER_B_ANGADA,
                        'arrayVersion': 19}
                    
                if isanila2:
                    list_of_params = {
                        'real_path': dict_of_real_paths[i][j],
                        'minDist': 15,
                        'param1': 70,
                        'param2': 20,
                        'minRadius': 15,
                        'maxRadius': 30,
                        'ptp': 80,
                        'tolerance': 8,
                        'detectFULL': True,
                        'cct': 2,
                        'range_type': 2,
                        'set_order_f':CircleDetect.SET_ORDER_F_ANILA,
                        'set_order_b':CircleDetect.SET_ORDER_B_ANILA,
                        'arrayVersion': 20}
                
                dict_list_of_params_folder[j] = list_of_params
            
            dict_list_of_params[i] = dict_list_of_params_folder
            
        dict_list_of_instances_list = {}

        for i in range(len(dict_of_real_paths)):
            dict_list_of_instances = {}
            for j in range(len(dict_of_real_paths[i])):
                my_instance = CircleDetectInstance(
                dict_list_of_params[i][j]['real_path'], 
                detectFULL=dict_list_of_params[i][j]['detectFULL'], 
                cct= dict_list_of_params[i][j]['cct'], 
                range_type=dict_list_of_params[i][j]['range_type'], 
                arrayVersion=dict_list_of_params[i][j]['arrayVersion'],
                minimumDist= dict_list_of_params[i][j]['minDist'], 
                parameter2=dict_list_of_params[i][j]['param2'], 
                minimumRadius= dict_list_of_params[i][j]['minRadius'], 
                maximumRadius=dict_list_of_params[i][j]['maxRadius'], 
                ptpthreshold=dict_list_of_params[i][j]['ptp'], 
                tolerance=dict_list_of_params[i][j]['tolerance'],
                set_order_f=dict_list_of_params[i][j]['set_order_f'],
                set_order_b=dict_list_of_params[i][j]['set_order_b'], 
                autoparam=True,
                single=True,
                isfront=True,
                )
            
                dict_list_of_instances[j] = my_instance
                
            dict_list_of_instances_list[i] = dict_list_of_instances
            
            print(self.path)
            
        dict_list_of_saves = {}

        dict_list_of_savepaths = {}

        for i in range(len(self.path)):
            # for j in range(len(dict_of_real_paths[i])):
            savepath = str(self.path[i]) + 'processed/'
            if os.path.exists(savepath) == False:
                os.mkdir(savepath)
                
            dict_list_of_savepaths[i] = savepath
                
            list_of_saves = []
            
            for j in range(len(dict_of_real_paths[i])):
                list_of_saves.append(savepath + 'processed_' + str(dict_of_dirs[i][j]))
                
            dict_list_of_saves[i] = list_of_saves
            
        for i in range(len(dict_list_of_instances_list)):
            f = None

            if os.path.isfile(dict_list_of_savepaths[i] + 'processed_img.txt') == False:
                f = open(dict_list_of_savepaths[i] + 'processed_img.txt', 'x')
                
            with open(dict_list_of_savepaths[i] + 'processed_img.txt', 'w') as f:
                for j in range(len(dict_list_of_instances_list[i])):
                    if dict_list_of_instances_list[i][j].cd != None:
                        vf, vb = dict_list_of_instances_list[i][j].cd.process_image(save=True, path=dict_list_of_saves[i][j])
                        f.write(str(dict_list_of_saves[i][j]) + '\n' + str(vf) + '\n' + str(vb) + '\n')
                        
            f.close()