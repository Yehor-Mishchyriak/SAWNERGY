# tmp:
import sys
sys.path.append("/Users/yehormishchyriak/Desktop/APA/AllostericPathwayAnalyzer/")

import unittest
import os
import allosteric_pathway_analyzer as apa

class AnalysesProcessorTestCase(unittest.TestCase):
    
    def setUp(self):
        self.ap = apa.analyses_processor()
        self.config = self.ap.cls_config

    def test_construct_csv_file_path(self):
        input_ =  dict(cpptraj_file_path="/path/to/the/1-2.dat", output_directory="/output/directory/")
        output = os.path.join("/output/directory/", self.cls_config["csv_file_name"].format(start=1, end=2))
        self.assertEqual(
             self.ap._construct_csv_file_path(**input_),
             output
        )

    def test_cpptraj_to_csv_incrementally(self):
        pass

    def test_cpptraj_to_csv_immediately(self):
        pass

    def test_process_target_directory(self):
        pass
