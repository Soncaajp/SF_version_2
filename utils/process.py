import os
import json
import cv2
import numpy as np
import datetime
import mxnet as mx
import mxboard
from utils.metrics import ROCAUC, AverageROCAUC, TripletAccuracy, TripletLoss
from time import time




def Affectnet_valence_metrics(config):
    metrics = dict()
    for data_type in ['train', 'val']:
        metrics[data_type] = [mx.metric.MAE(name='{}-mae'.format(data_type), output_names=['val_output'], label_names=['softmax_label']),
        mx.metric.MSE(name='{}-mse'.format(data_type), output_names=['val_output'], label_names=['softmax_label'])]
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    return train_metrics, val_metrics

def Affectnet_metrics(config, sm_out_id=2):
    metrics = dict()
    for data_type in ['train', 'val']:
        metrics[data_type] = [mx.metric.Accuracy(name='{}-acc'.format(data_type), output_names=['softmax_output'], label_names=['softmax_label']),
                              mx.metric.NegativeLogLikelihood(name='{}_cross_entropy'.format(data_type), output_names=['softmax_output'], label_names=['softmax_label'])]
        metrics[data_type].append(
            AverageROCAUC(n_classes=len(config['emotions_list']), out_id=sm_out_id, name='{}-mean_roc_auc'.format(data_type)))
        # for class_id in range(8):
        #     metrics[data_type].append(ROCAUC(name='{}-roc_auc_{}'.format(data_type, config['emotions_list'][class_id]), class_id=class_id))
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    return train_metrics, val_metrics


def FEC_metrics(config):
    metrics = dict()
    for data_type in ['train', 'val']:
        metrics[data_type] = ([TripletAccuracy(name='{}-triplet-acc-{}'.format(data_type, emb_size), out_id=i + len(config['emb_size'])) for i, emb_size in enumerate(config['emb_size'])] +
                              [TripletLoss(name='{}-triplet-loss-{}'.format(data_type, emb_size), out_id=i) for i, emb_size in enumerate(config['emb_size'])])
        # for class_id in range(8):
        #     metrics[data_type].append(ROCAUC(name='{}-roc_auc_{}'.format(data_type, config['emotions_list'][class_id]), class_id=class_id))
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    return train_metrics, val_metrics



def score_dir(module, val_iter, config):
    lr_schedule = mx.lr_scheduler.FactorScheduler(step=2000, factor=0.5, stop_factor_lr=1e-5)
    lr_schedule.base_lr = 1e-5
    module.init_optimizer(optimizer='adam', optimizer_params=(('lr_scheduler', lr_schedule),
                                                              # ('learning_rate', 1e-5),
                                                              # ('momentum', 0.9),
                                                              ('wd', 1e-6)
                                                              ))

    train_metrics, val_metrics = Affectnet_valence_metrics(config)
    save_model_prefix = config['save_model_prefix_template'].format(config['layers'], 'x'.join([str(a) for a in config['img_size']]))
    mxboard_dir = os.path.join('mxboard_logs',
                               str(datetime.datetime.now()).replace(' ', '_') + '_' + save_model_prefix.split('/')[-1])
    if not os.path.exists(mxboard_dir):
        os.makedirs(mxboard_dir)
    sw = mxboard.SummaryWriter(logdir=mxboard_dir, flush_secs=5)

    # train_metrics, val_metrics = Affectnet_metrics(config)

    val_iter.reset()

    # for metric in val_metrics:
    #     metric.reset()
    for batch_id, batch in enumerate(val_iter):
        module.forward(batch, is_train=False)
        # if batch_id == 0:
        preds = module.get_outputs()[0].asnumpy()
        preds = np.argmax(preds, axis=1)
        imgs = batch.data[0].asnumpy()
        # print('val imgs shape', imgs.shape)
        for i in range(len(imgs)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            img[:] = (np.moveaxis(imgs[i], 0, -1) * 255).astype(np.uint8)
            cv2.putText(img, config['emotions_list'][preds[i]], (10, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            imgs[i] = np.moveaxis(img.astype(float) / 255., -1, 0)
        # print(imgs.shape)
        sw.add_image('val_minibatch', imgs, 0)
        sw.add_image('val_minibatch', imgs, 1)
        for i, metric in enumerate(val_metrics):
            module.update_metric(metric, batch.label)
    for metric in val_metrics:
        print(metric.get())
        # sw.add_scalar(tag=metric.name, value=metric.get()[1], global_step=0)


def train_Affectnet(module, train_iter, val_iter, config):
    # lr_schedule = mx.lr_scheduler.FactorScheduler(step=2000, factor=0.5, stop_factor_lr=1e-5)
    # lr_schedule.base_lr = 1e-5
    module.init_optimizer(optimizer='sgd', optimizer_params=(#('lr_scheduler', lr_schedule),
                                                              ('learning_rate', 0.01),
                                                              # ('momentum', 0.9),
                                                              ('wd', 1e-6)
                                                              ))

    save_model_prefix = config['save_model_prefix']
    mxboard_dir = os.path.join('mxboard_logs',
                               str(datetime.datetime.now()).replace(' ', '_') + '_' + save_model_prefix.split('/')[-1])
    if not os.path.exists(mxboard_dir):
        os.makedirs(mxboard_dir)
    sw = mxboard.SummaryWriter(logdir=mxboard_dir, flush_secs=5)

    train_metrics, val_metrics = Affectnet_valence_metrics(config)

    # train_iter.n_objects = 30
    for epoch in range(config['load_epoch'], 1000):
        print('epoch = ', epoch)
        train_iter.reset()
        for metric in train_metrics:
            metric.reset()
        for batch_id, batch in enumerate(train_iter):
            if batch_id == 0:
                sw.add_image('train_minibatch', batch.data[0].asnumpy(), epoch)
            module.forward(batch, is_train=True)  # compute predictions
            for metric in train_metrics:
                module.update_metric(metric, batch.label)  # accumulate prediction accuracy
                # print(metric.get())
            module.backward()  # compute gradients
            module.update()  # update parameters
            if (batch_id > 0) and (batch_id % config['metric_update_period'] == 0):
                # print(train_metric.get())
                # print('Epoch {} batch {}\t{}'.format(epoch, batch_id, '\t'.join(m + ' ' + str(v) for m, v in zip(*train_metric.get()))))
                for metric in train_metrics:
                    sw.add_scalar(tag=metric.name, value=metric.get()[1], global_step=train_iter.global_num_inst)
                    metric.reset()

        module.save_checkpoint(save_model_prefix, epoch + 1)
        val_iter.reset()

        for metric in val_metrics:
            metric.reset()
        for batch_id, batch in enumerate(val_iter):
            module.forward(batch, is_train=False)
            if batch_id == 0:
                preds = module.get_outputs()[0].asnumpy()
                preds = np.argmax(preds, axis=1)
                imgs = batch.data[0].asnumpy()
                print('val imgs shape', imgs.shape)
                for i in range(len(imgs)):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                    img[:] = (np.moveaxis(imgs[i], 0, -1) * 255).astype(np.uint8)
                    #cv2.putText(img, config['emotions_list'][preds[i]], (10, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    imgs[i] = np.moveaxis(img.astype(float) / 255., -1, 0)
                sw.add_image('val_minibatch', imgs, epoch)
            for i, metric in enumerate(val_metrics):
                module.update_metric(metric, batch.label)
        for metric in val_metrics:
            sw.add_scalar(tag=metric.name, value=metric.get()[1], global_step=train_iter.global_num_inst)
