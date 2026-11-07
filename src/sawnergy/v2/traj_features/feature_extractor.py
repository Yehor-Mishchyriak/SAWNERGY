from __future__ import annotations

# built-in
from pathlib import Path
import logging
# local
from sawnergy.rin import rin_util
from sawnergy import sawnergy_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__) 

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class FeatureExtractor:

    __SUPPORTED_FEATURES: tuple = ()

    def __init__(self, cpptraj_path: Path | str | None = None):
        _logger.debug("Initializing RINBuilder with cpptraj_path=%s", cpptraj_path)
        if isinstance(cpptraj_path, str):
            cpptraj_path = Path(cpptraj_path)

        self.cpptraj = rin_util.locate_cpptraj(explicit=cpptraj_path, verify=True)
        _logger.info("Using cpptraj at %s", self.cpptraj)

    @classmethod
    def supported_features(cls):
        return cls.__SUPPORTED_FEATURES

    @classmethod
    def new_feature(cls, name: str, extractor: function):
        # if valid:
        cls.__setattr__(name, extractor)
        # else:
        # raise ...
        cls.__SUPPORTED_FEATURES += (name,)
