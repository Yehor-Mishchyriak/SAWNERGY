import icontract


@icontract.require(lambda kind: kind in ("pairwise", "global"), error=ValueError, enabled=True)
@icontract.require(lambda scope: scope in ("window", "trajectory"), error=ValueError, enabled=True)
def feature(*, kind: str, scope: str):
    """Mark a feature method without wrapping or executing it."""
    def decorate(method):
        method.feature_kind = kind
        method.feature_scope = scope
        return method

    return decorate


class FeatureExtractor:
    """Base class for subclass-defined trajectory feature extraction."""
