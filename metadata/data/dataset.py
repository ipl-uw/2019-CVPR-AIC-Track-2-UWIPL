import os
import sys
import json,ast
import random
import logging
import torch.utils.data as data

from .transformer import get_transformer, load_image

class BaseDataset(data.Dataset):
    def __init__(self, opt, data_type, id2rid):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.data_type = data_type
        self.dataset = self._load_data(opt.data_dir+ '/' + data_type + '/data.txt')
        self.id2rid = id2rid
        self.data_size = len(self.dataset)
        self.transformer = get_transformer(opt)

    def __getitem__(self, index):
        image_file, box, attr_ids = self.dataset[index % self.data_size]
        
        input = load_image(image_file, box, self.opt, self.transformer)
        #input = load_image(image_file, self.opt, self.transformer)

        # label
        labels = list()
        for index, attr_id in enumerate(attr_ids):
            labels.append(self.id2rid[index][attr_id])

        return input, labels

    def __len__(self):
        return self.data_size

    def _load_data(self, data_file):
        print(data_file)
        dataset = list()
        if not os.path.exists(data_file):
            return dataset
        with open(data_file) as d:
            for line in d.readlines():
                line = json.dumps(ast.literal_eval(line))
                dataset.append(self.readline(line))
        #import pdb; pdb.set_trace()
        if self.opt.shuffle:
            logging.info("Shuffle %s Data" %(self.data_type))
            random.shuffle(dataset)
        else:
            logging.info("Not Shuffle %s Data" %(self.data_type))
        return dataset
    
    def readline(self, line):
        vbrand_list = ['Dodge', 'Ford', 'Chevrolet', 'GMC', 'Honda', 'Chrysler', 'Jeep', 'Hyundai',\
                        'Subaru', 'Toyota', 'Buick', 'others', 'KIA', 'Nissan', 'Volkswagen',\
                        'Oldsmobile', 'BMW', 'Cadillac', 'Volvo', 'Pontiac', 'Mercury', 'Lexus',\
                        'Saturn', 'Benz', 'Mazda', 'Scion', 'RAM', 'Mini', 'Lincoln', 'Audi',\
                        'Mitsubishi']
        vtype_list = ['SUV', 'PickupTruck', 'Sedan', 'Minivan', 'Truck', 'Hatchback', 'Bus']
        vcolor_list = ['Black', 'White', 'Red', 'Gray', 'Silver', 'Blue', 'Gold', 'Green', 'Yellow']
        data = [None, None,None]
        #print(line)
        line = ast.literal_eval(line)
        line = ast.literal_eval(line)

        #line = json.loads(line)
        if "image_file" in line:
            data[0] = line["image_file"]
        if 'box' in line:
            data[1] = line["box"]
        if 'id' in line:
            data[2] = line["id"]
            vtype = data[2][0]
            vbrand = data[2][1]
            vcolor = data[2][2]

            if (vtype not in vtype_list) or (vbrand not in vbrand_list) or (vcolor not in vcolor_list):
                print(data[0],data[2])
        
        return data
