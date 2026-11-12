import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sawnergy.v2.traj_features.feature_extractor import FeatureExtractor, feature


_OUTPUT = dict(dtype="float64", units="1", statistic="mean")


@pytest.mark.parametrize("kind", ["pairwise", "global"])
@pytest.mark.parametrize("scope", ["window", "trajectory"])
def test_feature_marks_original_method_without_executing_it(kind, scope):
    def method(self, ctx):
        raise AssertionError("Decoration must not run the method")

    signature = inspect.signature(method)
    decorated = feature(kind=kind, scope=scope, shape=("N", "N") if kind == "pairwise" else (), **_OUTPUT)(method)

    assert decorated is method
    assert decorated.feature_kind == kind
    assert decorated.feature_scope == scope
    assert decorated.feature_requires == ()
    assert inspect.signature(decorated) == signature


def test_feature_preserves_instance_binding_and_return_value():
    class Extractor(FeatureExtractor):
        @feature(kind="global", shape=(), **_OUTPUT, scope="window")
        def values(self, ctx):
            return self, ctx

    first, second = Extractor(), Extractor()
    ctx = object()
    assert first.values(ctx) == (first, ctx)
    assert second.values(ctx) == (second, ctx)
    assert first.values.feature_kind == "global"
    assert first.values.feature_scope == "window"


def test_feature_preserves_method_exception():
    error = RuntimeError("extraction failed")

    class Extractor(FeatureExtractor):
        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="trajectory")
        def values(self, ctx):
            raise error

    with pytest.raises(RuntimeError) as caught:
        Extractor().values(object())
    assert caught.value is error


@pytest.mark.parametrize("parameter", ["kind", "scope"])
@pytest.mark.parametrize("invalid", ["unknown", "", None, 1, [], {}])
def test_feature_rejects_invalid_metadata(parameter, invalid):
    arguments = {"kind": "pairwise", "scope": "window", parameter: invalid}
    with pytest.raises(ValueError):
        feature(**arguments, shape=("N", "N"), **_OUTPUT)


def test_feature_validation_remains_active_under_optimization():
    source = """
from sawnergy.v2.traj_features.feature_extractor import FeatureExtractor, feature
_OUTPUT = dict(dtype="float64", units="1", statistic="mean")
for arguments in ({'kind': 'invalid', 'scope': 'window'},
                  {'kind': 'global', 'scope': 'invalid'},
                  {'kind': 'global', 'scope': 'window', 'requires': ['input']},
                  {'kind': 'global', 'scope': 'window', 'requires': ('input', 'input')}):
    try:
        feature(**arguments, shape=(), **_OUTPUT)
    except ValueError:
        continue
    raise RuntimeError('Invalid feature metadata was accepted')

for dependency in ('missing', 'summary', 'later'):
    class Extractor(FeatureExtractor):
        @feature(kind='global', shape=(), **_OUTPUT, scope='trajectory', requires=(dependency,))
        def summary(self, ctx):
            raise RuntimeError('Discovery executed a feature')

        @feature(kind='global', shape=(), **_OUTPUT, scope='window')
        def later(self, ctx):
            raise RuntimeError('Discovery executed a feature')

    try:
        Extractor()._feature_methods()
    except ValueError:
        continue
    raise RuntimeError('Invalid feature dependencies were accepted')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-O", "-c", source], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_discovery_preserves_declaration_order_and_binds_each_instance():
    class Extractor(FeatureExtractor):
        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def z_pairs(self, ctx):
            return self, ctx

        def helper(self):
            raise AssertionError("Discovery evaluated a helper")

        @property
        def unrelated(self):
            raise AssertionError("Discovery evaluated a property")

        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory")
        def a_global(self, ctx):
            raise AssertionError("Discovery ran a feature")

    first, second = Extractor(), Extractor()
    methods = first._feature_methods()
    assert list(methods) == ["z_pairs", "a_global"]
    assert all(method.__self__ is first for method in methods.values())
    assert methods["z_pairs"].__func__ is Extractor.z_pairs
    ctx = object()
    assert methods["z_pairs"](ctx) == (first, ctx)
    assert second._feature_methods()["z_pairs"].__self__ is second
    methods.clear()
    assert list(first._feature_methods()) == ["z_pairs", "a_global"]


def test_discovery_keeps_override_position_and_appends_new_features():
    class Parent(FeatureExtractor):
        def helper(self, ctx):
            pass

        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def first(self, ctx):
            pass

        @feature(kind="global", shape=(), **_OUTPUT, scope="window")
        def second(self, ctx):
            pass

    class Child(Parent):
        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory")
        def added(self, ctx):
            pass

        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="trajectory")
        def first(self, ctx):
            pass

        @feature(kind="global", shape=(), **_OUTPUT, scope="window")
        def helper(self, ctx):
            pass

    methods = Child()._feature_methods()
    assert list(methods) == ["first", "second", "added", "helper"]
    assert methods["first"].__func__ is Child.first
    assert methods["first"].feature_scope == "trajectory"
    assert methods["second"].__func__ is Parent.second
    assert list(Parent()._feature_methods()) == ["first", "second"]


@pytest.mark.parametrize("replacement", [lambda self, ctx: None, None, property(lambda self: 1)])
def test_discovery_removes_undecorated_overrides(replacement):
    class Parent(FeatureExtractor):
        @feature(kind="global", shape=(), **_OUTPUT, scope="window")
        def values(self, ctx):
            pass

    class Child(Parent):
        values = replacement

    assert Child()._feature_methods() == {}
    assert list(Parent()._feature_methods()) == ["values"]


def test_discovery_appends_feature_reintroduced_after_removal():
    class Parent(FeatureExtractor):
        @feature(kind="global", shape=(), **_OUTPUT, scope="window")
        def first(self, ctx):
            pass

        @feature(kind="global", shape=(), **_OUTPUT, scope="window")
        def second(self, ctx):
            pass

    class Child(Parent):
        def first(self, ctx):
            pass

    class Grandchild(Child):
        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory")
        def first(self, ctx):
            pass

    methods = Grandchild()._feature_methods()
    assert list(methods) == ["second", "first"]
    assert methods["first"].__func__ is Grandchild.first


def test_discovery_follows_mro_for_diamond_inheritance():
    class Root(FeatureExtractor):
        @feature(kind="global", shape=(), **_OUTPUT, scope="window")
        def values(self, ctx):
            pass

    class Left(Root):
        pass

    class Right(Root):
        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory")
        def values(self, ctx):
            pass

    class Diamond(Left, Right):
        pass

    assert Diamond()._feature_methods()["values"].__func__ is Right.values

    class Suppressed(Left, Right):
        def values(self, ctx):
            pass

    assert Suppressed()._feature_methods() == {}


def test_discovery_empty_base():
    assert FeatureExtractor()._feature_methods() == {}


@pytest.mark.parametrize("requires", [(), ("first",), ("second", "first")])
def test_feature_records_explicit_dependencies(requires):
    def method(self, ctx):
        raise AssertionError("Decoration executed a feature")

    decorated = feature(kind="global", shape=(), **_OUTPUT, scope="trajectory", requires=requires)(method)
    assert decorated is method
    assert decorated.feature_requires == requires


@pytest.mark.parametrize("requires", ["first", ["first"], None, 1,
                                     ("first", "first"), ("",), (1,), ([],)])
def test_feature_rejects_invalid_dependency_declarations(requires):
    with pytest.raises(ValueError):
        feature(kind="global", shape=(), **_OUTPUT, scope="window", requires=requires)


def test_discovery_accepts_multiple_earlier_dependencies_across_scopes():
    class Extractor(FeatureExtractor):
        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def first(self, ctx):
            raise AssertionError("Discovery executed a feature")

        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="trajectory", requires=("first",))
        def second(self, ctx):
            raise AssertionError("Discovery executed a feature")

        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory", requires=("second", "first"))
        def summary(self, ctx):
            raise AssertionError("Discovery executed a feature")

    extractor = Extractor()
    methods = extractor._feature_methods()
    assert list(methods) == ["first", "second", "summary"]
    assert methods["summary"].feature_requires == ("second", "first")
    assert all(method.__self__ is extractor for method in methods.values())


@pytest.mark.parametrize("dependency", ["missing", "summary", "later", "helper"])
def test_discovery_rejects_unavailable_dependencies(dependency):
    class Extractor(FeatureExtractor):
        def helper(self, ctx):
            pass

        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory", requires=(dependency,))
        def summary(self, ctx):
            raise AssertionError("Discovery executed a feature")

        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def later(self, ctx):
            raise AssertionError("Discovery executed a feature")

    with pytest.raises(ValueError, match=f"summary.*{dependency}"):
        Extractor()._feature_methods()


def test_discovery_revalidates_dependencies_when_inherited_feature_is_removed():
    class Parent(FeatureExtractor):
        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def first(self, ctx):
            pass

        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory", requires=("first",))
        def summary(self, ctx):
            pass

    class Child(Parent):
        def first(self, ctx):
            pass

    class Grandchild(Child):
        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def first(self, ctx):
            pass

    for cls in (Child, Grandchild):
        with pytest.raises(ValueError, match="summary.*first"):
            cls()._feature_methods()
    assert list(Parent()._feature_methods()) == ["first", "summary"]


@pytest.mark.parametrize("dependency", ["first", "later"])
def test_discovery_validates_dependencies_after_decorated_override(dependency):
    class Parent(FeatureExtractor):
        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def first(self, ctx):
            pass

        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory", requires=("missing",))
        def summary(self, ctx):
            pass

    class Child(Parent):
        @feature(kind="pairwise", shape=("N", "N"), **_OUTPUT, scope="window")
        def later(self, ctx):
            pass

        @feature(kind="global", shape=(), **_OUTPUT, scope="trajectory", requires=(dependency,))
        def summary(self, ctx):
            pass

    if dependency == "later":
        with pytest.raises(ValueError, match="summary.*later"):
            Child()._feature_methods()
    else:
        methods = Child()._feature_methods()
        assert list(methods) == ["first", "summary", "later"]
        assert methods["summary"].__func__ is Child.summary
    with pytest.raises(ValueError, match="summary.*missing"):
        Parent()._feature_methods()
