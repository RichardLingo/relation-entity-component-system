"""RECS 后端注册表。

提供后端注册、获取与自动检测功能。
默认后端为 'numpy'（而非 'auto'），确保现有代码零改动。
"""

from .base import BackendBase
from .numpy_backend import NumpyBackend

# 始终可用的后端
_NAMED_BACKENDS: dict[str, BackendBase] = {}
_NAMED_BACKENDS['numpy'] = NumpyBackend()


def _try_register(name: str, module_path: str, class_name: str) -> None:
    """惰性注册可选后端。依赖缺失时静默跳过。"""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        _NAMED_BACKENDS[name] = cls()
    except ImportError:
        pass


# 惰性注册可选后端（仅在对应包已安装时生效）
_try_register('torch', 'recs.backends.torch_backend', 'TorchBackend')
_try_register('mlx', 'recs.backends.mlx_backend', 'MLXBackend')
_try_register('jax', 'recs.backends.jax_backend', 'JAXBackend')
_try_register('tf', 'recs.backends.tf_backend', 'TFBackend')


def get_backend(name: str = 'numpy') -> BackendBase:
    """获取后端实例。

    默认值为 'numpy'（而非 'auto'），确保现有代码零改动。
    'auto' 时自动选择最优可用后端。

    Args:
        name: 后端名称，支持 'numpy' / 'torch' / 'mlx' / 'jax' / 'tf' / 'auto'。

    Returns:
        对应的后端实例。

    Raises:
        ValueError: 当指定后端不可用时抛出。
    """
    if name == 'auto':
        for candidate in ['mlx', 'torch', 'jax', 'numpy']:
            if candidate in _NAMED_BACKENDS:
                return _NAMED_BACKENDS[candidate]
    if name not in _NAMED_BACKENDS:
        available = list(_NAMED_BACKENDS.keys())
        raise ValueError(f"后端 '{name}' 不可用。可用: {available}")
    return _NAMED_BACKENDS[name]


def list_backends() -> list[str]:
    """返回所有已注册的后端名称列表。"""
    return list(_NAMED_BACKENDS.keys())
