"""Vendored upstream architectures, used only to export ONNX at build time.

Nothing here is imported by the running bot — the runtime loads .onnx files and
never constructs a torch module. See tools/export_onnx.py.
"""
from .rrdbnet_arch import RRDBNet

__all__ = ["RRDBNet"]
