"""Embedded history compression and evidence recovery for Python agents."""
__version__ = '0.13.1'
from .action_contract import ActionContract, RequiredAction
from .context_plugin import ContextPlugin
from .memory import HistoryArchive
from .memory_agent import DeepSeekChat, run_agent
from .provenance import HistoricalWriteGuard
from .write_transform import FreshWriteGuard, WriteTransform

__all__ = ["ActionContract", "RequiredAction", "ContextPlugin", "HistoryArchive", "DeepSeekChat", "run_agent", "HistoricalWriteGuard", "FreshWriteGuard", "WriteTransform"]
