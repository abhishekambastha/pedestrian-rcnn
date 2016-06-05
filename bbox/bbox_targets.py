import numpy as np
import cfg
from bbox.cython_bbox import bbox_overlaps

def add_bbox_regression_targets(roidb):
    num_images = len(roidb)
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for idx in xrange(num_images):
        rois = roidb[idx]['boxes']
        max_overlaps = roidb[idx]['max_overlaps']
        max_classes = roidb[idx]['max_classes']
        roidb[idx]['bbox_targets'] = _compute_targets(rois, max_overlaps, max_classes)

        means = np.tile(
            np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1))
        stds = np.tile(
            np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes, 1))

        print 'means'
        print means
        print 'stds'
        print stds

        print 'Normalizing targets'
        for idx in xrange(num_images):
            targets = roidb[idx]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                roidb[idx]['bbox_targets'] -= means[cls, :]
                roidb[idx]['bbox_targets'] /= stds[cls, :]

        return means.ravel(), stds.ravel()

def _compute_targets(rois, overlaps, labels):
    gt_inds = np.where(overlaps == 1)

    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)

    ex_gt_overlaps = bbox_overlaps( \
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float), \
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)

    return targets

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets
