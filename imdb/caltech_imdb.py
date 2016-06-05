import numpy as np
import os
import json
import cPickle
import scipy.sparse
from imdb import imdb

class caltech_imdb(imdb):

    def __init__(self, image_set, path=None):
        imdb.__init__(self, image_set)
        self._data_path = path
        self._image_set = image_set
        self._classes = ('__background__', 'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self.cache_path = './'


    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'Images', index + self._image_ext)
        return image_path


    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index


    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '[INFO - imdb]: {} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        data = self._get_annotations_json()
        gt_roidb = [self._load_caltech_annotation(index, data) for index in self._image_index]

        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)
        print '[INFO - imdb]: wrote gt roidb to {}'.format(cache_file)
        return gt_roidb


    def _get_annotations_json(self):
        filename = os.path.join(self._data_path, 'Annotations', 'annotations.json')
        with open(filename) as f:
            data = json.load(f)

        return data


    def _load_caltech_annotation(self, index, data):
        objs = data[index]
        num_objs = data[index]['num_objects']

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs['coords_list']):
            x1 = float(obj['x1'])
            y1 = float(obj['y1'])
            x2 = float(obj['x2'])
            y2 = float(obj['y2'])

            cls = self._class_to_ind['person']
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)


        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas':seg_areas}
