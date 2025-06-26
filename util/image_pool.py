# util/image_pool.py

import torch

class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def query(self, images):
        # During inference, this is never used.
        return images
