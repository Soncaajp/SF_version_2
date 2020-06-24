from utils.dataiter import AffectnetIter, FECIter
from utils.process import train_Affectnet, score_dir
import fmobilefacenet
import mxnet as mx

def run_Affectnet_training():
    config = {'batch_size': 64,
              'val_batch_size': 40,
              'img_size': (112, 112),  # (128, 128),
              'metric_update_period': 50,
              'layers': 50,
              'load_epoch': 0,
              #'load_path': '/media/nlab/data/test/resnext50-valence-llr',
              'save_model_prefix': '/media/nlab/data/SF/mbnet-singleframe',
              'emotions_list': ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt'],
              # 'multiply_basic_ratio': 4
              }
    train_iter = AffectnetIter(data_json_path='../training.csv',
                               batch_size=config['batch_size'], train=True, img_size=config['img_size'], detector = None)

    train_iter.global_num_inst = int(train_iter.n_objects / config['batch_size']) * config['batch_size'] * config['load_epoch']

    fc1 = fmobilefacenet.get_symbol()
    module = mx.mod.Module(fc1, context=mx.gpu(0))
    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(arg_params=None, aux_params=None, initializer=mx.init.MSRAPrelu(),
                       allow_missing=True)

    val_iter = AffectnetIter(data_json_path='../validation.csv',
                              batch_size=config['val_batch_size'], train=False, img_size=config['img_size'], detector = None)
    train_Affectnet(module, train_iter, val_iter, config)


if __name__ == '__main__':
    run_Affectnet_training()

# /media/nlab/data/AffectNet_Aligned/