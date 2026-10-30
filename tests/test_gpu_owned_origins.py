"""Pure-CPU origin mutations; no installed framework is imported or executed."""

import hashlib
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from transformer import gpu_owned_origins as origin


@pytest.fixture(params=["pytorch", "mlx"])
def fixture(tmp_path, request):
    name = request.param
    root = tmp_path.resolve()
    base = root / name
    package = base / ("torch" if name == "pytorch" else "mlx")
    package.mkdir(parents=True)
    payload, files, modules = {}, {}, {}

    def file(relative, content=b"authenticated fixture file"):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        payload[str(path)] = sha
        files[path.relative_to(root).as_posix()] = sha
        return path

    def module(key, path=None):
        result = ModuleType(key)
        if path is not None:
            result.__file__ = str(path)
            result.__spec__ = ModuleSpec(key, None, origin=str(path))
        modules[key] = result
        return result

    if name == "pytorch":
        top = module("torch", file("torch/__init__.py"))
        core = module("torch._C", file("torch/_C.so"))
        top._C = core
        module("torch.nn", file("torch/nn/__init__.py"))
        module("torch.optim", file("torch/optim/__init__.py"))
        for label, class_name, singleton in (
            ("_ops", "_Ops", "ops"),
            ("_classes", "_Classes", "classes"),
        ):
            backing_name = "torch." + label
            content = (
                "from types import ModuleType\n"
                f"class {class_name}(ModuleType):\n"
                f"    __file__ = '{label}.py'\n"
                "    def __init__(self):\n"
                f"        super().__init__('torch.{singleton}')\n"
                f"{singleton} = {class_name}()\n"
            ).encode()
            path = file("torch/" + label + ".py", content)
            backing = module(backing_name, path)
            exec(compile(content, str(path), "exec"), vars(backing))
            synthetic = vars(backing)[singleton]
            setattr(top, singleton, synthetic)
            modules["torch." + singleton] = synthetic
        native = module("torch._C._dynamo")
        core._dynamo = native
        native.eval_frame = module("torch._C._dynamo.eval_frame")
        module("torchgen", file("torchgen/__init__.py"))
        module("functorch", file("functorch/__init__.py"))
        locations = [str(package / "__init__.py"), str(package)]
    else:
        top = module("mlx")
        top.__path__ = [str(package)]
        top.__spec__ = ModuleSpec("mlx", None, is_package=True)
        top.__spec__.submodule_search_locations = [str(package)]
        core = module("mlx.core", file("mlx/core.so"))
        top.core = core
        module("mlx.nn", file("mlx/nn/__init__.py"))
        module("mlx.optimizers", file("mlx/optimizers/__init__.py"))
        core.fast = module("mlx.core.fast")
        locations = [str(package)]
    inventory = {
        "files": files,
        "layout": {"packages": {"torch" if name == "pytorch" else "mlx": locations}},
    }
    return SimpleNamespace(
        name=name,
        root=root,
        base=base,
        package=package,
        payload=payload,
        inventory=inventory,
        modules=modules,
        file=file,
        module=module,
        top=top,
        core=core,
    )


def verify(f):
    return origin.verify_loaded_framework_modules(f.name, f.payload, f.inventory, f.root, f.modules)


def test_complete_real_file_synthetic_and_native_alias_origins(fixture):
    result = verify(fixture)
    assert result["module_count"] == len(fixture.modules)
    assert set(result) == {
        "schema",
        "framework",
        "module_origins",
        "synthetic_origins",
        "native_aliases",
        "namespace_packages",
        "module_count",
    }
    assert result["native_aliases"]
    if fixture.name == "pytorch":
        assert set(result["synthetic_origins"]) == {"torch.ops", "torch.classes"}
    else:
        assert result["namespace_packages"]["mlx"]["locations"] == [str(fixture.package)]


@pytest.mark.parametrize("suffix", ["nn", "optim"])
def test_foreign_loaded_submodule_cannot_use_authenticated_root(fixture, suffix):
    name = (
        ("torch." + suffix)
        if fixture.name == "pytorch"
        else ("mlx.nn" if suffix == "nn" else "mlx.optimizers")
    )
    foreign = fixture.root / "foreign.py"
    foreign.write_text("# alternate implementation\n")
    fixture.modules[name].__file__ = str(foreign)
    with pytest.raises(ValueError, match="payload"):
        verify(fixture)


@pytest.mark.parametrize("mutation", ["bytes", "inventory", "spec", "relative", "symlink"])
def test_real_module_origin_mutations_fail(fixture, mutation):
    key = "torch.nn" if fixture.name == "pytorch" else "mlx.nn"
    module = fixture.modules[key]
    path = Path(module.__file__)
    if mutation == "bytes":
        path.write_text("changed\n")
    elif mutation == "inventory":
        del fixture.inventory["files"][path.relative_to(fixture.root).as_posix()]
    elif mutation == "spec":
        module.__spec__.origin = "built-in"
    elif mutation == "relative":
        module.__file__ = "nn/__init__.py"
    else:
        alias = fixture.root / "alias.py"
        alias.symlink_to(path)
        module.__file__ = str(alias)
    with pytest.raises(ValueError):
        verify(fixture)


def test_unexplained_fileless_module_rejected(fixture):
    fixture.module(("torch" if fixture.name == "pytorch" else "mlx") + ".unexplained")
    with pytest.raises(ValueError, match="Unexplained fileless"):
        verify(fixture)


def test_missing_module_object_rejected(fixture):
    fixture.modules[("torch" if fixture.name == "pytorch" else "mlx") + ".nn"] = None
    with pytest.raises(ValueError, match="Missing framework module"):
        verify(fixture)


def test_native_core_top_level_alias_identity_required(fixture):
    setattr(fixture.top, "_C" if fixture.name == "pytorch" else "core", ModuleType("other"))
    with pytest.raises(ValueError, match="extension alias"):
        verify(fixture)


def test_native_submodule_cannot_reuse_an_unrelated_object(fixture):
    key = "torch._C._dynamo.eval_frame" if fixture.name == "pytorch" else "mlx.core.fast"
    fixture.modules[key] = ModuleType(key)
    with pytest.raises(ValueError, match="submodule identity"):
        verify(fixture)


def test_native_alias_name_is_not_a_blanket_fileless_exemption(fixture):
    fixture.module("torch._C.unbound" if fixture.name == "pytorch" else "mlx.core.unbound")
    with pytest.raises(ValueError, match="submodule identity"):
        verify(fixture)


def test_native_alias_name_must_match(fixture):
    key = "torch._C._dynamo.eval_frame" if fixture.name == "pytorch" else "mlx.core.fast"
    fixture.modules[key].__name__ = "unrelated"
    with pytest.raises(ValueError, match="submodule name"):
        verify(fixture)


@pytest.mark.parametrize("fixture", ["pytorch"], indirect=True)
@pytest.mark.parametrize("mutation", ["singleton", "class", "constructor", "spec", "label"])
def test_torch_synthetic_namespace_is_exactly_authenticated(fixture, mutation):
    module = fixture.modules["torch.ops"]
    cls = type(module)
    if mutation == "singleton":
        fixture.top.ops = ModuleType("torch.ops")
    elif mutation == "class":

        class Substitute(cls):
            pass

        fixture.modules["torch.ops"] = Substitute()
    elif mutation == "constructor":
        cls.__init__ = lambda self: None
    elif mutation == "spec":
        module.__spec__ = ModuleSpec("torch.ops", None, origin="built-in")
    else:
        module.__file__ = "unexpected.py"
    with pytest.raises(ValueError, match="(namespace|constructor)"):
        verify(fixture)


@pytest.mark.parametrize("fixture", ["mlx"], indirect=True)
@pytest.mark.parametrize(
    "mutation", ["extra", "spec_only", "no_members", "init_file", "missing_inventory"]
)
def test_mlx_namespace_paths_are_exact(fixture, mutation):
    if mutation == "extra":
        fixture.top.__path__.append(str(fixture.root))
    elif mutation == "spec_only":
        fixture.top.__spec__.submodule_search_locations = [str(fixture.root)]
    elif mutation == "no_members":
        fixture.top.__path__ = [str(fixture.root / "empty")]
        fixture.top.__spec__.submodule_search_locations = fixture.top.__path__
        fixture.inventory["layout"]["packages"]["mlx"] = fixture.top.__path__
        (fixture.root / "empty").mkdir()
    elif mutation == "init_file":
        fixture.file("mlx/__init__.py")
    else:
        del fixture.inventory["files"]["mlx/mlx/optimizers/__init__.py"]
    with pytest.raises(ValueError):
        verify(fixture)


def test_no_framework_import_is_needed():
    # The implementation uses the explicitly supplied mapping, never imports the
    # selected framework to repair a missing origin or inspect a native object.
    before = {key for key in sys.modules if key in {"torch", "mlx.core"}}
    with pytest.raises(ValueError, match="Empty framework origin payload"):
        origin.verify_loaded_framework_modules("pytorch", {}, {"files": {}}, "/", {})
    assert {key for key in sys.modules if key in {"torch", "mlx.core"}} == before


def test_native_alias_with_core_file_still_requires_object_identity(fixture):
    key = "torch._C._dynamo.eval_frame" if fixture.name == "pytorch" else "mlx.core.fast"
    fixture.modules[key] = ModuleType(key)
    fixture.modules[key].__file__ = fixture.core.__file__
    with pytest.raises(ValueError, match="submodule identity"):
        verify(fixture)


def test_actual_native_alias_can_share_its_extension_file(fixture):
    key = "torch._C._dynamo.eval_frame" if fixture.name == "pytorch" else "mlx.core.fast"
    fixture.modules[key].__file__ = fixture.core.__file__
    assert key in verify(fixture)["native_aliases"]
