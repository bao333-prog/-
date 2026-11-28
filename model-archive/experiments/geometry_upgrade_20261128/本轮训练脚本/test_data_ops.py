"""确认高分辨率缓存来自源图，ROI使用源图坐标，绝不偷偷放大旧缓存。"""
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

class DataOpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.folder=Path(self.tmp.name)
        image=np.random.default_rng(11).integers(0,256,(480,640,3),dtype=np.uint8)
        self.source=self.folder/'source.png';Image.fromarray(image).save(self.source)
        self.digest=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.row=dict(source=str(self.source),family_id=self.digest,kind='original',environment='target',split='train')

    def tearDown(self):self.tmp.cleanup()

    def test_320_reads_original_and_differs_from_224_upscale(self):
        from data_ops import render_row
        import sys
        sys.path.insert(0,str(Path(__file__).parent/'source_snapshot'))
        from oven_food.augment import prepare_image
        output,meta=render_row(self.row,320)
        with Image.open(self.source) as im:
            expected=prepare_image(im,image_size=320,resize_mode='letterbox')
            old=prepare_image(im,image_size=224,resize_mode='letterbox')
        self.assertTrue(np.array_equal(output,expected))
        self.assertFalse(np.array_equal(output,prepare_image(old,image_size=320,resize_mode='letterbox')))
        self.assertEqual(meta['cache_source_size'],[640,480]);self.assertFalse(meta['source_already_low_resolution'])

    def test_roi_uses_original_coordinates(self):
        from data_ops import render_row
        row={**self.row,'kind':'annotation_roi','crop_xyxy':[100,110,500,310]}
        output,meta=render_row(row,320)
        self.assertEqual(output.size,(320,320));self.assertEqual(meta['cache_crop_xyxy'],[100,110,500,310])
        self.assertEqual(meta['cache_source_size'],[640,480])

    def test_source_fingerprint_mismatch_refused(self):
        from data_ops import render_row
        with self.assertRaises(ValueError):render_row({**self.row,'family_id':'bad'},320)

    def test_invalid_roi_not_clamped_silently(self):
        from data_ops import render_row
        with self.assertRaises(ValueError):render_row({**self.row,'kind':'annotation_roi','crop_xyxy':[0,0,800,900]},320)

if __name__=='__main__':unittest.main()
