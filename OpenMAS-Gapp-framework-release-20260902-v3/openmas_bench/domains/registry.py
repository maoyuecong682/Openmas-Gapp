from __future__ import annotations

from .base import DomainContext, DomainPlugin
from .bbh import BBHDomainPlugin
from .financebench import FinanceBenchDomainPlugin
from .finqa import FinQADomainPlugin
from .scibench import SciBenchDomainPlugin


_PLUGINS = (
    BBHDomainPlugin(),
    FinanceBenchDomainPlugin(),
    SciBenchDomainPlugin(),
    FinQADomainPlugin(),
)


def get_domain_plugin(dataset_id: str, metric_name: str = "") -> DomainPlugin | None:
    for plugin in _PLUGINS:
        if dataset_id in plugin.dataset_ids or (metric_name and metric_name in plugin.metric_names):
            return plugin
    return None


def registered_domain_plugins() -> tuple[DomainPlugin, ...]:
    return _PLUGINS


__all__ = ["DomainContext", "DomainPlugin", "get_domain_plugin", "registered_domain_plugins"]

