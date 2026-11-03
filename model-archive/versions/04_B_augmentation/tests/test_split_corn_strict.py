"""Catch leakage, unreviewed admission and byte-changing exports."""
import importlib.util
import sys
import unittest
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
SPEC = importlib.util.find_spec('split_corn_strict')
if SPEC:
    import split_corn_strict
    from split_corn_strict import assign_splits, validate_splits


class StrictSplitTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC, 'Strict reviewed-group export has not been implemented')

    def row(self, identifier, group, digest, status='approved'):
        return dict(id=identifier, source_group_id=group, sha256=digest,
                    collection_review_status=status)

    def test_same_group_photos_follow_one_split(self):
        rows = [self.row('a', 'shoot1', 'x'), self.row('b', 'shoot1', 'y')]
        result = assign_splits(rows, {'shoot1':'train'})
        self.assertEqual([r['split'] for r in result], ['train', 'train'])

    def test_pending_and_excluded_never_enter_split(self):
        rows = [self.row('a','g','x','pending_review'), self.row('b','g','y','excluded')]
        self.assertEqual([r['split'] for r in assign_splits(rows, {'g':'train'})], ['', ''])

    def test_unplanned_approved_group_stops_export(self):
        with self.assertRaises(ValueError):
            assign_splits([self.row('a','new_group','x')], {})

    def test_cross_split_group_is_rejected(self):
        rows = [dict(self.row('a','g','x'), split='train'),
                dict(self.row('b','g','y'), split='test')]
        with self.assertRaises(ValueError):
            validate_splits(rows)

    def test_cross_split_duplicate_bytes_are_rejected(self):
        rows = [dict(self.row('a','g1','same'), split='train'),
                dict(self.row('b','g2','same'), split='val')]
        with self.assertRaises(ValueError):
            validate_splits(rows)

    def test_unreviewed_assigned_image_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits([dict(self.row('a','g','x','pending_review'), split='train')])

    def test_numeric_source_metadata_does_not_break_author_extraction(self):
        # Commons also returns numeric GPS metadata alongside the textual fields.
        extractor = getattr(split_corn_strict, 'plain_metadata', None)
        self.assertIsNotNone(extractor, 'Mixed-type metadata normalization is missing')
        result = extractor({'Artist':{'value':'<b>Alice</b>'}, 'GPSLatitude':{'value':27.5}})
        self.assertEqual(result['Artist'], 'Alice')
        self.assertEqual(result['GPSLatitude'], '27.5')


ARTIFACT = ROOT/'data/corn_strict_split_2026-10-20'


@unittest.skipUnless((ARTIFACT/'review.csv').exists(), 'Export not built')
class StrictExportTest(unittest.TestCase):
    def read(self, name):
        with (ARTIFACT/name).open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))

    def test_split_manifests_image_paths_resolve_in_delivered_package(self):
        for split in ('train','val','test'):
            for row in self.read(split+'.csv'):
                with self.subTest(id=row['id']):
                    self.assertTrue((ARTIFACT/row['image_path']).is_file())

    def test_live_restored_download_link_is_not_still_marked_missing(self):
        for row in self.read('review.csv'):
            if row['source_checked_date'] and row['download_url'].startswith('http'):
                with self.subTest(id=row['id']):
                    self.assertNotIn('download_url', row['metadata_missing'].split(';'))


if __name__ == '__main__':
    unittest.main()
