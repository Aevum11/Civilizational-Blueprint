"""
main.py
ET32 Bridge — top-level entry shim for PyCharm / direct Python invocation.

Running this file is equivalent to running et32_bridge_main.py.
The authoritative entry point with argument parsing, logging, and the full
startup sequence is et32_bridge_main.  This shim exists so that PyCharm's
green-button runner and any script that does `python main.py` work correctly.

Derived from P ∘ D ∘ T = E.
"""

import sys
import os

# Ensure the package root is on the path when run directly
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from et32_bridge_main import main  # noqa: E402  (import after path fix)

if __name__ == "__main__":
    sys.exit(main())