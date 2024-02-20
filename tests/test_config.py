# test_config.py

import unittest
import iQuanta.config


class TestFraziphy(unittest.TestCase):
    def test_list_int(self):
        
        data = [1, 2, 3]
        self.assertEqual(data, [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
