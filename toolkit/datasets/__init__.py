from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from .ptbtir import PTBTIRDataset
from .LSOTB import LSOTBDataset
from .VOTTIR2015 import VOTTIRDataset

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB100' == name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAV' == name:
            dataset = UAVDataset(**kwargs)
        elif 'GOT_10k' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'PTBTIR' == name:
            dataset = PTBTIRDataset(**kwargs)
        elif 'LSOTB' == name:
            dataset = LSOTBDataset(**kwargs)
        elif 'VOTTIR2015' == name or "VOTTIR2017" == name:
            dataset = VOTTIRDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

