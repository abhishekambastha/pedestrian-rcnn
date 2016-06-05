import numpy as np
import imdb
import sys
from roidb import roidb
from imdb.caltech_imdb import caltech_imdb

#Prepare Dataset
imDb = caltec_imdb('train', './data')
roidb.prepare_roidb(imDb)
roidb.filter_roidb(imDb.roidb)




def init_caffe():
    pass


def main():
    init_caffe()
    imdb, roidb = combined_roidb('imdb_name')
    train('solver.prototxt', roidb, 'output_dir',
          pretrained_model='VGG16', max_iters='40000')

def train(solver, roidb, output_dir, pretrained_model, max_iters):
    print 'Solver {}'.format(solver)
    print 'Roidb'
    print 'Pretrained Model is {}'.format(pretrained_model)
    print 'Max Iters {}'.format(max_iters)
