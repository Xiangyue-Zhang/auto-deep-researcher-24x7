"""AutoResearcher Core - Autonomous ML Experiment Agent Framework."""

from .execution import ExecutionBackend, LocalExecutionBackend, SSHExecutionBackend, build_execution_backend
from .loop import ResearchLoop
from .memory import MemoryManager
from .monitor import ExperimentMonitor
from .agents import AgentDispatcher
from .tools import ToolRegistry

__version__ = "0.1.1"
__all__ = [
    "AgentDispatcher",
    "ExecutionBackend",
    "ExperimentMonitor",
    "LocalExecutionBackend",
    "MemoryManager",
    "ResearchLoop",
    "SSHExecutionBackend",
    "ToolRegistry",
    "build_execution_backend",
]
