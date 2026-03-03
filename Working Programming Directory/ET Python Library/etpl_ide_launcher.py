#!/usr/bin/env python3
"""
ETPL IDE Launcher - Integrated Stages 1 + 2
Complete IDE with all enhancements
Usage: python etpl_ide_launcher.py
"""

import sys
import os

# Ensure ETPL.py is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication

# Import Stage 1 base IDE
from etpl_ide_stage1 import ETPLIDEMainWindow

# Import Stage 2 integration
from etpl_ide_stage2_enhancements import integrate_stage2_into_ide


def main():
    """
    Launch ETPL IDE with all Stage 1 + Stage 2 features.
    Derived from ET: T agency initiating the complete system
    """
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("ETPL IDE - Complete Edition")
    app.setOrganizationName("Exception Theory")
    
    # Create base IDE (Stage 1)
    print("Initializing ETPL IDE...")
    print("Stage 1: Base IDE framework...")
    window = ETPLIDEMainWindow()
    
    # Integrate Stage 2 enhancements
    print("Stage 2: Adding enhanced features...")
    print("  - Error detection system")
    print("  - Code intelligence")
    print("  - Enhanced import tracing")
    print("  - Graph visualization")
    window = integrate_stage2_into_ide(window)
    
    # Show window
    print("IDE ready!")
    print("\nFeatures available:")
    print("  ✓ ETPL syntax highlighting")
    print("  ✓ Parse, interpret, compile (F5/F6/F7)")
    print("  ✓ Real-time error detection")
    print("  ✓ Code completion")
    print("  ✓ Import dependency graph")
    print("  ✓ Self-hosting package builder")
    print("\nRefer to ETPL_IDE_ROADMAP.md for complete feature list.")
    
    window.show()
    
    # Display welcome message in console
    window.console.append_info("=" * 60)
    window.console.append_success("ETPL IDE - Exception Theory Programming Language")
    window.console.append_info("Stages 1 + 2 Integrated - Full Feature Set")
    window.console.append_info("=" * 60)
    window.console.append_output("Quick Start:")
    window.console.append_output("  1. File → New (Ctrl+N) to create .pdt file")
    window.console.append_output("  2. Write ETPL code using symbol shortcuts")
    window.console.append_output("  3. Parse (F5), Interpret (F6), or Compile (F7)")
    window.console.append_output("  4. View errors in Errors panel (real-time)")
    window.console.append_output("  5. Tools → Trace Import Chain for dependencies")
    window.console.append_info("=" * 60)
    window.console.append_output("ET Symbol Shortcuts:")
    window.console.append_output("  Ctrl+O = ∘ (binding)")
    window.console.append_output("  Ctrl+R = → (arrow)")
    window.console.append_output("  Ctrl+L = λ (lambda)")
    window.console.append_output("  Ctrl+I = ∞ (infinity)")
    window.console.append_output("  Ctrl+P = ψ (quantum)")
    window.console.append_info("=" * 60)
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
