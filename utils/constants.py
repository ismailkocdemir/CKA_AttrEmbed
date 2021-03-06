import os

import utils.io as io

class Constants(io.JsonSerializableClass):
    def __init__(self):
        pass

class ExpConstants(Constants):
    def __init__(
            self,
            exp_name='default_exp',
            out_base_dir=os.path.join(
                os.getcwd(),'temp_exp/')):
        self.exp_name = exp_name
        self.out_base_dir = out_base_dir
        self.exp_dir = os.path.join(self.out_base_dir,self.exp_name)
        self.milestones = [60, 120, 160]
        self.warmup_epoch = 1 
        self.num_epochs = 200


def save_constants(name_const_dict,outdir):
    for name, const in name_const_dict.items():
        print(f'Saving {name} constants ...')
        filename = os.path.join(outdir,f'{name}_constants.json')
        const.to_json(filename)
