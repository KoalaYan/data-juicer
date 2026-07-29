import importlib
import os

from .base_op import OPERATORS


_LAZY_IMPORT_ENV = "DATA_JUICER_LAZY_OP_IMPORT"
_OP_SUFFIX_TO_PACKAGE = (
    ("_mapper", "mapper"),
    ("_filter", "filter"),
    ("_deduplicator", "deduplicator"),
    ("_selector", "selector"),
    ("_grouper", "grouper"),
    ("_aggregator", "aggregator"),
)


def lazy_op_import_enabled():
    return os.environ.get(_LAZY_IMPORT_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def ensure_operator_registered(op_name):
    """Import one conventionally named operator module when lazy mode is on."""
    if op_name in OPERATORS.modules:
        return
    if not lazy_op_import_enabled():
        raise KeyError(f"Operator is not registered: {op_name}")
    package = next(
        (
            package
            for suffix, package in _OP_SUFFIX_TO_PACKAGE
            if op_name.endswith(suffix)
        ),
        None,
    )
    if package is None:
        raise KeyError(
            f"Cannot infer operator package from name: {op_name}"
        )
    try:
        importlib.import_module(f"data_juicer.ops.{package}.{op_name}")
    except ModuleNotFoundError as error:
        expected = f"data_juicer.ops.{package}.{op_name}"
        if error.name != expected:
            raise
        raise KeyError(
            f"Operator module does not exist: {expected}"
        ) from error
    if op_name not in OPERATORS.modules:
        raise KeyError(
            f"Module loaded but did not register operator: {op_name}"
        )


def load_ops(process_list, op_env_manager=None):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :param op_env_manager: The OPEnvManager to try to merge environment specs of different OPs that have common
        dependencies. Only available when min_common_dep_num_to_combine >= 0.
    :return: The op instance list.
    """
    ops = []
    new_process_list = []

    for process in process_list:
        op_name, args = list(process.items())[0]
        ensure_operator_registered(op_name)
        ops.append(OPERATORS.modules[op_name](**args))
        new_process_list.append(process)

    # store the OP configs into each OP
    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg

    # update op runtime environment if OPEnvManager is enabled
    if op_env_manager:
        # first round: record and merge possible common env specs
        for op in ops:
            op_name = op._name
            op_env_spec = op.get_env_spec()
            op_env_manager.record_op_env_spec(op_name, op_env_spec)
        # second round: update op runtime environment
        for op in ops:
            op_name = op._name
            op_env_spec = op_env_manager.get_op_env_spec(op_name)
            op._requirements = op_env_spec.pip_pkgs
            # if the runtime_env is not set for this OP, update the runtime_env as well
            if op.runtime_env is None:
                op.runtime_env = op_env_spec.to_dict()

    return ops
