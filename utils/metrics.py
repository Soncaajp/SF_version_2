import mxnet as mx
import mxboard
import numpy as np
from sklearn.metrics import roc_auc_score

class MxboardAccuracy(mx.metric.Accuracy):
    def __init__(self, logdir, **kwargs):
        self.num_inst = 0
        self.global_num_inst = 0
        self.sw = mxboard.SummaryWriter(logdir=logdir#, flush_secs=5
                                )
        super(MxboardAccuracy, self).__init__(**kwargs)

    def reset(self):
        print(self.name, self.num_inst)
        self.global_num_inst += self.num_inst
        if 'val' in self.name:
            self.sw.add_scalar(tag=self.name, value=self.get(), global_step=self.global_num_inst)
        super(MxboardAccuracy, self).reset()



class TripletAccuracy(mx.metric.EvalMetric):
    def __init__(self, out_id, **kwargs):
        super(TripletAccuracy, self).__init__(**kwargs)
        self.out_id = out_id

    def update(self, thresholds, emdeddings):
        emb = emdeddings[self.out_id].asnumpy()
        batch_size = int(emb.shape[0] / 3)
        dists = np.array([np.linalg.norm(emb[batch_size:2 * batch_size] - emb[2 * batch_size:], axis=1),
                          np.linalg.norm(emb[batch_size:2 * batch_size] - emb[:batch_size], axis=1),
                          np.linalg.norm(emb[:batch_size] - emb[2 * batch_size:], axis=1)])
        # emb = emb / np.sqrt(np.sum(np.square(emb), axis=1, keepdims=True))
        # dists = np.array([np.arccos(0.99999 * np.sum(emb[batch_size:2 * batch_size] * emb[2 * batch_size:], axis=1)),
        #                   np.arccos(0.99999 * np.sum(emb[batch_size:2 * batch_size] * emb[:batch_size], axis=1)),
        #                   np.arccos(0.99999 * np.sum(emb[:batch_size] * emb[2 * batch_size:], axis=1))])
        # dists = np.array([np.sum(emb[batch_size:2 * batch_size] * emb[2 * batch_size:], axis=1),
        #                   np.sum(emb[batch_size:2 * batch_size] * emb[:batch_size], axis=1),
        #                   np.sum(emb[:batch_size] * emb[2 * batch_size:], axis=1)])
        # print('emb id:', self.out_id - len(emdeddings) // 2, '\tembeddings:', emb.shape)
        # print(emb.T)
        # print('dists:', dists.shape)
        # print(dists.T)
        nearest = np.argmin(dists, axis=0)

        self.sum_metric += (nearest == 0).sum()
        self.num_inst += len(nearest)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0


class TripletLoss(mx.metric.EvalMetric):
    def __init__(self, out_id, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)
        self.out_id = out_id

    def update(self, thresholds, emdeddings):
        bce = emdeddings[self.out_id].asnumpy()

        # print('emb id:', self.out_id, '\tbce:', bce.shape)
        # print(bce.T)

        self.sum_metric += bce.sum()
        self.num_inst += len(bce)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

class ROCAUC(mx.metric.EvalMetric):
    def __init__(self, class_id, **kwargs):
        super(ROCAUC, self).__init__(**kwargs)
        self.class_id = class_id

    def update(self, labels, preds):
        label = labels[0].asnumpy().ravel() == self.class_id
        pred = preds[0].asnumpy()[:, self.class_id]
        self.labels.extend(label)
        self.probas.extend(pred)
        # print(label, preds[0].asnumpy())
        self.sum_metric = self.num_inst * roc_auc_score(self.labels, self.probas)
        self.num_inst += len(label)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.labels = []
        self.probas = []
        self.num_inst = 0
        self.sum_metric = 0.0

class AverageROCAUC(mx.metric.EvalMetric):
    def __init__(self, n_classes, out_id=0, labels_id=0, **kwargs):
        super(AverageROCAUC, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.out_id = out_id
        self.labels_id = labels_id

    def update(self, labels, preds):
        label = labels[self.labels_id].asnumpy().ravel()
        pred = preds[self.out_id].asnumpy()
        self.labels.extend(label)
        self.probas.extend(pred)
        # print(label, preds[self.out_id].asnumpy())
        self.num_inst += len(label)
        self.sum_metric = 0
        for class_id in range(self.n_classes):
            # print(class_id, np.array(self.labels) == class_id,
            #                                  np.array(self.probas)[:, class_id])
            self.sum_metric += (float(self.num_inst) / self.n_classes *
                               roc_auc_score(np.array(self.labels) == class_id,
                                             np.array(self.probas)[:, class_id]))
        pass

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.labels = []
        self.probas = []
        self.num_inst = 0
        self.sum_metric = 0.0