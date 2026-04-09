"""AutoResearcher Core - Autonomous ML Experiment Agent Framework."""

from .loop import ResearchLoop
from .memory import MemoryManager
from .monitor import ExperimentMonitor
from .agents import AgentDispatcher
from .tools import ToolRegistry

__version__ = "0.1.1"
__all__ = ["ResearchLoop", "MemoryManager", "ExperimentMonitor", "AgentDispatcher", "ToolRegistry"]
