import numpy as np
import PIL.Image
from imdb import imdb
from cfg import cfg

def get_training_roidb(imdb):
    'Removing flipped images'
    print 'Preparing training data...'
    prepare_roidb(imdb)
    print 'done'
    return imdb.roidb

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

##After solver call
def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def valid_bbox(entry):
        height = entry.get('height')
        width = entry.get('width')
        boxes = entry.get('boxes')
        valid = True
        for box in boxes:
            if box[0] > width or box[2] > width:
                valid = False
                break
            if box[1] > height or box[3] > height:
                valid = False
                break

        return valid

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry.get('max_overlaps')
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        if valid:
            valid = valid_bbox(entry)
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry) and valid_bbox(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


if __name__ == '__main__':
    from imdb.caltech_imdb import caltech_imdb
    imdb = caltech_imdb('train', './data')
