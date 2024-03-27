from .uavdark import UAVDARKDataset
from .darktrack import DARKTRACKDataset


datapath = {
            'UAVDark135':'',
            'DarkTrack2021':'',
            }

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):

        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAVDark' in name:
            dataset = UAVDARKDataset(**kwargs)
        elif 'DarkTrack2021' in name:
            dataset = DARKTRACKDataset(**kwargs)
        
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset
