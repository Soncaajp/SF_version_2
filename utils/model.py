import mxnet as mx
import numpy as np

def get_resnext_softmax(config, train_iter):
    # if config['load_epoch'] == 0:
    #     load_path = 'pretrained_models/' + 'resnext-{}_224x224_FEC_256'.format(config['layers'])
    # else:
    #     load_path = 'models/' + 'resnext-oneface-{}_{}'.format(config['layers'], 'x'.join([str(a) for a in config['img_size']]))
    print('loading model {}, epoch {}'.format(config['load_path'], config['load_epoch']))
    embedding, arg_params, aux_params = mx.model.load_checkpoint(config['load_path'], config['load_epoch'])

    embedding = embedding.get_internals()['activation0_output']

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')

    features = embedding(data=data)
    # features = mx.sym.L2Normalization(mx.sym.FullyConnected(features, num_hidden=256, name='emb_256'), name='l2_norm')


    # # features = mx.sym.Activation(features, act_type='tanh')
    # # features = mx.sym.Dropout(features, p=0.4)
    # features = mx.sym.FullyConnected(features, num_hidden=512, name='fc_1')
    # features = mx.sym.BatchNorm(features)
    # features = mx.sym.Activation(features, act_type='relu')
    # # features = mx.sym.Dropout(features, p=0.25)
    # # features = mx.sym.FullyConnected(features, num_hidden=1024, name='fc_1')
    # # features = mx.sym.Activation(features, act_type='tanh')
    # # features = mx.sym.BatchNorm(features)
    # # features = mx.sym.Dropout(features, p=0.4)
    # # features = mx.sym.FullyConnected(features, num_hidden=train_iter.n_classes, name='fc_2')

    # # features = mx.sym.FullyConnected(features, num_hidden=train_iter.n_classes, name='logits')
    out = mx.sym.Pooling(features, kernel = (2,2), pool_type ='avg')
    out = mx.sym.FullyConnected(out, num_hidden=1, name='valence_fc')
    out = mx.sym.LinearRegressionOutput(out, label, name = 'val')


    # out = mx.sym.SoftmaxOutput(features, label=speaker, name='softmax')

    # softmax_probas = mx.sym.MakeLoss(mx.sym.BlockGrad(mx.sym.softmax(features, axis=1)), name='softmax_probas')
    # loss = mx.sym.Group([out, softmax_probas])

    # mx.viz.plot_network(out, node_attrs={"shape":"oval","fixedsize":"false"})
    # plt.show()

    module = mx.mod.Module(out, context=mx.gpu(0))

    # import logging
    # head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(level=logging.DEBUG, format=head)

    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(arg_params=arg_params, aux_params=aux_params, initializer=mx.init.MSRAPrelu(),
                       allow_missing=True)
    return module


def l2_triplet(features, thresholds, config, labels=None):
    far = mx.sym.slice_axis(features, axis=0, begin=0, end=config['batch_size'], name='slice_far')
    near1 = mx.sym.slice_axis(features, axis=0, begin=config['batch_size'], end=2 * config['batch_size'], name='slice_near1')
    near2 = mx.sym.slice_axis(features, axis=0, begin=2 * config['batch_size'], end=3 * config['batch_size'], name='slice_near2')
    thresholds = mx.sym.slice_axis(thresholds, axis=0, begin=0, end=config['batch_size'], name='slice_thresholds')

    dist_fn1 = mx.sym.sum(mx.sym.square(far - near1), axis=1)
    dist_fn2 = mx.sym.sum(mx.sym.square(far - near2), axis=1)
    dist_n12 = mx.sym.sum(mx.sym.square(near1 - near2), axis=1)

    triplet_loss = (mx.sym.relu(dist_n12 - dist_fn1 + thresholds) +
                    mx.sym.relu(dist_n12 - dist_fn2 + thresholds))

    if labels is not None:
        triplet_loss = triplet_loss * (mx.sym.slice_axis(labels, axis=0, begin=0, end=config['batch_size']) == -1)

    triplet_loss = mx.sym.MakeLoss(triplet_loss, grad_scale=config['triplet_gs'])
    return triplet_loss


def angle_triplet(features, thresholds, config, labels=None):
    features = mx.sym.L2Normalization(features, name='l2_norm')

    far = mx.sym.slice_axis(features, axis=0, begin=0, end=config['batch_size'])
    near1 = mx.sym.slice_axis(features, axis=0, begin=config['batch_size'], end=2 * config['batch_size'])
    near2 = mx.sym.slice_axis(features, axis=0, begin=2 * config['batch_size'], end=3 * config['batch_size'])
    thresholds = mx.sym.slice_axis(thresholds, axis=0, begin=0, end=config['batch_size'])

    dist_fn1 = mx.sym.arccos(0.999999999 * mx.sym.sum(far * near1, axis=1))
    dist_fn2 = mx.sym.arccos(0.999999999 * mx.sym.sum(far * near2, axis=1))
    dist_n12 = mx.sym.arccos(0.999999999 * mx.sym.sum(near1 * near2, axis=1))

    triplet_loss = (mx.sym.relu(dist_n12 - dist_fn1 + thresholds) +
                    mx.sym.relu(dist_n12 - dist_fn2 + thresholds))
    # triplet_loss = (dist_n12 + 0 * thresholds)
    if labels is not None:
        triplet_loss = triplet_loss * (mx.sym.slice_axis(labels, axis=0, begin=0, end=config['batch_size']) == -1)

    triplet_loss = mx.sym.MakeLoss(triplet_loss)
    return triplet_loss


def get_resnext_triplet(config, train_iter):
    if config['load_epoch'] == 0:
        embedding, arg_params, aux_params = mx.model.load_checkpoint(config['load_model_path'], config['load_epoch'])
            # 'pretrained_models/' + 'resnext-FEC-{}_224x224'.format(config['layers']), config['load_epoch'])
    else:
        embedding, arg_params, aux_params = mx.model.load_checkpoint(config['load_model_path'], config['load_epoch'])
            # 'models/' + 'resnext-FEC-{}_{}'.format(config['layers'], 'x'.join([str(a) for a in config['img_size']])),

    embedding = embedding.get_internals()['flatten0_output']

    data = mx.sym.Variable('data')
    thresholds = mx.sym.Variable('thresholds')

    features = embedding(data=data)

    #
    # features = mx.sym.Dropout(features, p=0.4)
    # features = mx.sym.FullyConnected(features, num_hidden=512, name='fc_1')
    # features = mx.sym.BatchNorm(features)
    # features = mx.sym.Activation(features, act_type='relu')
    # features = mx.sym.Dropout(features, p=0.5)
    # features = mx.sym.FullyConnected(features, num_hidden=train_iter.n_classes, name='logits')
    # out = mx.sym.SoftmaxOutput(features, label=speaker, name='softmax')
    #
    # module = mx.mod.Module(out, context=mx.gpu(0))

    embeddings = []
    triplet_losses = []
    for dim in config['emb_size']:

        emb = mx.sym.FullyConnected(features, num_hidden=dim, name='emb_{}'.format(dim))
        emb = mx.sym.L2Normalization(emb, name='l2_norm')
        # norm = np.sqrt(dim)
        # emb = emb / norm
        triplet_losses.append(l2_triplet(emb, thresholds, config))
        embeddings.append(mx.sym.MakeLoss(mx.sym.BlockGrad(emb)))

    # features = mx.sym.FullyConnected(features, num_hidden=256, name='low_dim')
    # triplet_loss = l2_triplet(features, thresholds, config)
    loss = mx.sym.Group(triplet_losses + embeddings)

    module = mx.mod.Module(loss, context=mx.gpu(0), data_names=['data'], label_names=['thresholds'])

    # import logging
    # head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(level=logging.DEBUG, format=head)

    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(arg_params=arg_params, aux_params=aux_params, initializer=mx.init.MSRAPrelu(),
                       allow_missing=True)
    return module

def get_resnext_triplet_sm(config, train_iter):
    if config['load_epoch'] == 0:
        embedding, arg_params, aux_params = mx.model.load_checkpoint(
            'pretrained_models/' + 'resnext-{}_224x224_FEC_256'.format(config['layers']), 0)
    else:
        embedding, arg_params, aux_params = mx.model.load_checkpoint(
            'models/' + 'resnext-{}_{}-FEC_Affectnet-oneface-rotated-mbr4'.format(config['layers'], 'x'.join([str(a) for a in config['img_size']])),
            config['load_epoch'])

    embedding = embedding.get_internals()['flatten0_output']

    data = mx.sym.Variable('data')
    thresholds = mx.sym.Variable('thresholds')
    label = mx.sym.Variable('softmax_label')

    features = embedding(data=data)
    features = mx.sym.Dropout(features, p=0.4)

    embeddings = []
    triplet_losses = []
    for dim in config['emb_size']:
        emb = mx.sym.FullyConnected(features, num_hidden=dim, name='emb_{}'.format(dim))
        emb = mx.sym.L2Normalization(emb, name='l2_norm')
        # norm = np.sqrt(dim)
        # emb = emb / norm
        triplet_losses.append(l2_triplet(emb, thresholds, config))
        embeddings.append(mx.sym.MakeLoss(mx.sym.BlockGrad(emb)))

    assert len(config['emb_size']) == 1, 'which embeddings to use for softmax head'
    features = emb  # mx.sym.BlockGrad(emb)

    # features = mx.sym.Dropout(features, p=0.4)
    features = mx.sym.FullyConnected(features, num_hidden=512, name='fc_1')
    features = mx.sym.BatchNorm(features)
    features = mx.sym.Activation(features, act_type='relu')
    features = mx.sym.Dropout(features, p=0.3)

    sm_logits = mx.sym.FullyConnected(features, num_hidden=len(config['emotions_list']), name='logits')
    sm_logits = mx.sym.broadcast_mul(sm_logits, mx.sym.reshape(label, (-1, 1)) != -1)
    sm_out = mx.sym.SoftmaxOutput(sm_logits, label=label, name='softmax', ignore_label=-1, grad_scale=1.)
    # softmax_probas = mx.sym.MakeLoss(mx.sym.BlockGrad(mx.sym.softmax(sm_logits, axis=1)), name='softmax_probas')


    loss = mx.sym.Group(triplet_losses + embeddings + [sm_out])

    module = mx.mod.Module(loss, context=mx.gpu(0), data_names=['data'], label_names=['softmax_label', 'thresholds'])
    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(arg_params=arg_params, aux_params=aux_params, initializer=mx.init.MSRAPrelu(),
                       allow_missing=True)
    return module
