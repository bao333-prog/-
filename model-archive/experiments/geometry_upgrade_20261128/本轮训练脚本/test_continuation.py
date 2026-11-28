"""覆盖20轮上升、平台、下降，以及扩展段的停止规则。"""
import unittest
from continuation import should_stop

def history(counts):
    return [{'validation_accuracy':c/193} for c in counts]

class StopPolicyTests(unittest.TestCase):
    def test_no_early_stop(self): self.assertFalse(should_stop(history([180]*19)))
    def test_flat_at_20_stops(self): self.assertTrue(should_stop(history([180]*17+[190]*3)))
    def test_rising_at_20_continues(self): self.assertFalse(should_stop(history([180]*17+[188,189,190])))
    def test_turn_down_at_20_stops(self): self.assertTrue(should_stop(history([180]*17+[188,190,189])))
    def test_extension_waits_three_without_new_accuracy_high(self):
        h=[180]*17+[188,189,190]
        self.assertFalse(should_stop(history(h+[190,190])))
        self.assertTrue(should_stop(history(h+[190,190,190])))
        self.assertFalse(should_stop(history(h+[190,191,191])))
    def test_maximum_stops(self): self.assertTrue(should_stop(history(list(range(60)))))
    def test_extension_has_three_rounds_even_with_earlier_peak(self):
        h=[180]*9+[192]+[180]*7+[188,189,190]
        self.assertFalse(should_stop(history(h)))
        self.assertFalse(should_stop(history(h+[190])))
        self.assertFalse(should_stop(history(h+[190,190])))
        self.assertTrue(should_stop(history(h+[190,190,190])))

if __name__=='__main__': unittest.main()
