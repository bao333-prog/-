"""真实像素测试：保留分支必须不再插值；另一分支仍提供既有几何变化。"""
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).parent/'source_snapshot'))
from oven_food.augment import ProxyAugmenter, prepare_image


class PreserveTests(unittest.TestCase):
    def test_identity_branch_keeps_every_prepared_pixel(self):
        image=Image.fromarray(np.random.default_rng(2).integers(0,256,(126,224,3),dtype=np.uint8))
        augment=ProxyAugmenter('oven_preserve',image_size=224,seed=1,resize_mode='letterbox')
        self.assertTrue(np.array_equal(np.asarray(augment(image)),np.asarray(prepare_image(image,image_size=224,resize_mode='letterbox'))))

    def test_fixed_seed_reproduces_and_both_branches_exist(self):
        image=Image.fromarray(np.random.default_rng(3).integers(0,256,(224,224,3),dtype=np.uint8))
        a=ProxyAugmenter('oven_preserve',image_size=224,seed=42,resize_mode='letterbox')
        b=ProxyAugmenter('oven_preserve',image_size=224,seed=42,resize_mode='letterbox')
        outputs=[np.asarray(a(image)) for _ in range(30)]
        self.assertTrue(all(np.array_equal(x,np.asarray(b(image))) for x in outputs))
        identities=sum(np.array_equal(x,np.asarray(image)) for x in outputs)
        self.assertGreater(identities,0);self.assertLess(identities,30)

    def test_center_crop_refused(self):
        with self.assertRaises(ValueError):ProxyAugmenter('oven_preserve',resize_mode='center_crop')


if __name__=='__main__':unittest.main()
