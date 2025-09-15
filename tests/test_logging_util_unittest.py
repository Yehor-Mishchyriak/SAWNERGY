import unittest
import logging
import tempfile
from pathlib import Path
from tests.util_import import load_module

# Load as a simple top-level module
logging_util = load_module("logging_util", "logging_util.py")

class TestLoggingUtil(unittest.TestCase):
    def setUp(self):
        # Reset root handlers to make tests deterministic
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

    def test_configure_logging_creates_file_and_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            logging_util.configure_logging(logs_dir)
            root = logging.getLogger()
            self.assertTrue(root.handlers, "Root logger should have handlers")
            files = list(logs_dir.glob("sawnergy_*.log"))
            self.assertTrue(files, "Expected a sawnergy_*.log file")

    def test_configure_logging_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            logging_util.configure_logging(logs_dir)
            before = len(logging.getLogger().handlers)
            logging_util.configure_logging(logs_dir)
            after = len(logging.getLogger().handlers)
            self.assertEqual(before, after, "Should not double-register handlers")

if __name__ == "__main__":
    unittest.main()
