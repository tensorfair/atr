import sys
import os
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
from atr.libs.model import *


class ATR:
    def __init__(self):
        self.location = build_log_dir('training')
        fontgroups_labels = {'antiqua': 0, 'bastarda': 1, 'fraktur': 2, 'gotico_antiqua': 3, 'greek': 4, 'hebrew': 5, 'italic': 6, 'rotunda': 7, 'schwabacher': 8, 'textura': 9, '-': 10, 'not_a_font': 11}
        tw_labels = {'ma00131': 0, 'ma00967': 1, 'ma02771': 2, 'ma07721': 3, 'ma07487': 4, 'ma07488': 5, 'ma04614': 6, 'ma07718': 7}
        afgr_specs = {'dir': "fontgroupsdataset", 'batch_size': 32, 'x_size': 64, 'y_size': 64, 'epochs': 200, 'lr': 0.001, 'labels': fontgroups_labels}
        atr_specs = {'dir': "twdataset", 'batch_size': 32, 'x_size': 64, 'y_size': 64, 'epochs': 10, 'lr': 0.0005, 'labels': tw_labels}
        self.afgr = DenseNet121(afgr_specs, self.location, classes=12)
        self.atr = DenseNet121(atr_specs, self.location, classes=8, pretrained=True)
        self.afgr.train()
        self.atr.train()
        self.atr.freeze()


if __name__ == "__main__":
    ATR()

tf.logging.set_verbosity(old_v)
