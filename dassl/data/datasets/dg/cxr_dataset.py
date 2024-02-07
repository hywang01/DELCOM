import glob
import os.path as osp
import numpy as np
import pandas as pd
from ast import literal_eval

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

import pdb
from tqdm import tqdm


@DATASET_REGISTRY.register()
class CxrDG(DatasetBase):
    """CXR-DG (27 GB of .npy files).

    It contains 3 chest x-ray datasets:
        - NIH CX14: The number of images is 91,777 after filtering.
        - CheXpert: The number of images is 191,229 after filtering.
        - MIMIC-CXR: The number of images is 243,326 after filtering.

    """

    # root_path = '/media/userdisk0'
    dataset_dir = 'CXR_DG'  # TODO
    domains = ['nih', 'chxp', 'mimic', 'cha']

    finding_name = ['Atelectasis',
                    'Cardiomegaly',
                    'Consolidation',
                    'Edema',
                    'Effusion',
                    'Pneumonia',
                    'Pneumothorax',
                    'No Finding']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))  # full path of root
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            print('assign root path')

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )  # names of domains in cfg must be matched up with the domain list here

        # list of data instance, including:
        #         imgpath (str): image path
        #         label (int): class label
        #         domain (int): domain label.
        train = self.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, 'train'
        )
        val = self.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, 'val'
        )
        test = self.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS, 'test'
        )

        # ll = [train, val, test]
        # for i in range(len(ll)):
        #     temp = []
        #     for j in range(len(ll[i])):
        #         temp.append(ll[i][j]._label.shape[0])
        #     temp_set = set(temp)
        #     print(len(temp), temp_set)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_data(dataset_dir, input_domains, split):
        finding_names = ['Atelectasis',
                'Cardiomegaly',
                'Consolidation',
                'Edema',
                'Effusion',
                'Pneumonia',
                'Pneumothorax',
                'No Finding']

        def _load_data_from_directory(directory):

            # items_ is a list of paths of .npy files
            items_ = glob.glob(osp.join(directory, '*.npy'))

            return items_
        
        def _load_data_from_csv(directory):

            # items_ is a list of path-label pairs
            items_ = pd.read_csv(directory, converters={'label': literal_eval})
            items_ = items_.apply(lambda x: tuple(x), axis=1).values.tolist()

            return items_
        
        def _label_to_class_name(label, finding_names):
            
            class_names = []
            if np.all(label==0):
                class_names = 'No Finding'
            else:
                class_names = [f_str for f_str, l_n in zip(finding_names, label)
                               if l_n == 1]
                class_names = '_'.join(class_names)

            return class_names
        
        items = []

        for domain, dname in enumerate(input_domains):
            # domain is an index (int)
            # dname is a string

            # Example:
            #   dataset_dir = '/media/userdisk0/CXR_DG'
            #   dname = 'nih'
            #   train_dir = '/media/userdisk0/CXR_DG/nih/train'

            split_dir = osp.join(dataset_dir, dname, split)
            csv_dir = osp.join(dataset_dir, dname, dname+'_'+split+'.csv')
            impath_label_list = _load_data_from_csv(csv_dir)
            print(dname+' '+split)

            for img_name, label in tqdm(impath_label_list):
                # load_data = np.load(img_label_pair, allow_pickle=True).item()
                # impath = img_label_pair
                # label = load_data['label']
                # 
                impath = osp.join(split_dir, img_name)
                class_name = _label_to_class_name(label, finding_names)
                label = np.array(label)
                item = Datum(impath=impath, label=label, domain=domain, classname=class_name)
                items.append(item)
                #pdb.set_trace()

        return items
