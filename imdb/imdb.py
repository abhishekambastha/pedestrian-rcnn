class imdb(object):

    def __init__(self, name):
        self._name = name
        self._classes = []
        self._image_index = None
        self._roidb = None
        self.roidb_handler = self.gt_roidb

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def image_index(self):
        return self._image_index

    @property
    def name(self):
        return self._name

    @property
    def num_images(self):
        return len(self.image_index)

    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb

        self._roidb = self.roidb_handler()
        return self._roidb
