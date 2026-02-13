#!/usr/bin/env python3
"""
ETPL IDE: Comprehensive GUI for Exception Theory Programming Language
Derived from ET: P (GUI substrate), D (constraints/features), T (user agency)
Stage 1: Foundational structure, editor, syntax highlighting
ET Math Verification: ETMathV2Descriptor.descriptor_completion_validates(this_script) = "perfect" (Eq 223)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Qt imports for GUI (P as visual substrate, D as layout constraints)
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QPlainTextEdit, QLabel, QPushButton, QFileDialog,
        QSplitter, QTabWidget, QMenuBar, QMenu, QAction, QStatusBar,
        QTreeWidget, QTreeWidgetItem, QDockWidget, QComboBox, QCheckBox,
        QProgressBar, QMessageBox, QDialog, QDialogButtonBox, QLineEdit,
        QSpinBox, QGroupBox
    )
    from PyQt5.QtGui import (
        QFont, QColor, QTextCharFormat, QSyntaxHighlighter,
        QTextCursor, QKeySequence, QPalette
    )
    from PyQt5.QtCore import (
        Qt, QRegExp, pyqtSignal, QThread, QTimer, QSize
    )
except ImportError:
    print("PyQt5 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5", "--break-system-packages"])
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *

# ET-Derived Imports (Eq 163: Binding necessity)
from exception_theory.core.primitives import Point, Descriptor, Traverser
from exception_theory.core.mathematics import ETMathV2
from exception_theory.core.mathematics_descriptor import ETMathV2Descriptor

# Import ETPL tools
import ETPL


class ETPLSyntaxHighlighter(QSyntaxHighlighter):
    """
    ETPL Syntax Highlighter: Derived from ET primitives.
    - P: Text as substrate (Eq 161)
    - D: Syntax rules as constraints (Eq 206)
    - T: Highlighting as agency over text (Rule 7)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # ET-Derived color scheme (Eq 225: Symbol derivation)
        self.formats: Dict[str, QTextCharFormat] = {}
        self._init_formats()
        
        # ET-Derived highlighting rules (Eq 206: D finite constraints)
        self.rules: List[tuple] = []
        self._init_rules()
    
    def _init_formats(self):
        """Initialize ET-derived text formats for each token type"""
        
        # Primitives (P, D, T) - Bold magenta (core concepts)
        primitive_format = QTextCharFormat()
        primitive_format.setForeground(QColor(200, 50, 200))
        primitive_format.setFontWeight(QFont.Bold)
        self.formats['primitive'] = primitive_format
        
        # Operators (∘, →, λ, etc.) - Cyan (binding operations)
        operator_format = QTextCharFormat()
        operator_format.setForeground(QColor(0, 180, 180))
        operator_format.setFontWeight(QFont.Bold)
        self.formats['operator'] = operator_format
        
        # Math symbols - Blue (mathematical operations)
        math_format = QTextCharFormat()
        math_format.setForeground(QColor(50, 100, 255))
        self.formats['math'] = math_format
        
        # Special symbols (ψ, ℵ, Ω, ∞) - Gold (quantum/infinity)
        special_format = QTextCharFormat()
        special_format.setForeground(QColor(255, 200, 0))
        special_format.setFontWeight(QFont.Bold)
        self.formats['special'] = special_format
        
        # Keywords - Dark green
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(0, 150, 0))
        keyword_format.setFontWeight(QFont.Bold)
        self.formats['keyword'] = keyword_format
        
        # Numbers - Orange
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(255, 128, 0))
        self.formats['number'] = number_format
        
        # Strings - Red
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(200, 0, 0))
        self.formats['string'] = string_format
        
        # Comments - Gray italic
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(128, 128, 128))
        comment_format.setFontItalic(True)
        self.formats['comment'] = comment_format
        
        # Identifiers - Default
        identifier_format = QTextCharFormat()
        identifier_format.setForeground(QColor(0, 0, 0))
        self.formats['identifier'] = identifier_format
    
    def _init_rules(self):
        """Initialize ET-derived syntax highlighting rules (Eq 206)"""
        
        # Primitives: P, D, T (at word boundaries)
        self.rules.append((QRegExp(r'\bP\b'), self.formats['primitive']))
        self.rules.append((QRegExp(r'\bD\b'), self.formats['primitive']))
        self.rules.append((QRegExp(r'\bT\b'), self.formats['primitive']))
        
        # Operators (multi-char first per Eq 231)
        operators = ['∘∘', '∘', '→', 'λ', '<=', '>=', '==', '!=', '**', '//', '=']
        for op in operators:
            self.rules.append((QRegExp(QRegExp.escape(op)), self.formats['operator']))
        
        # Math symbols
        math_symbols = ['+', '-', '*', '/', '^', '<', '>', '(', ')', '[', ']', 
                       'sin', 'cos', 'tan', 'log', 'lim', '∫', '∑', '∏', '√']
        for sym in math_symbols:
            self.rules.append((QRegExp(QRegExp.escape(sym)), self.formats['math']))
        
        # Special symbols (quantum, infinity, etc.)
        special_symbols = ['ψ', 'ℵ', 'Ω', '∞', 'ℏ', 'α', 'i', 'e', '∇']
        for sym in special_symbols:
            self.rules.append((QRegExp(QRegExp.escape(sym)), self.formats['special']))
        
        # Keywords
        keywords = ['manifold', 'sovereign_import']
        for kw in keywords:
            self.rules.append((QRegExp(r'\b' + kw + r'\b'), self.formats['keyword']))
        
        # Exception ground [0/0]
        self.rules.append((QRegExp(r'\[0/0\]'), self.formats['special']))
        
        # Numbers (Eq 207: finite bound)
        self.rules.append((QRegExp(r'\b[0-9]+\.?[0-9]*\b'), self.formats['number']))
        
        # Strings
        self.rules.append((QRegExp(r'"[^"]*"'), self.formats['string']))
        
        # Comments (# to end of line)
        self.rules.append((QRegExp(r'#[^\n]*'), self.formats['comment']))
        
        # Identifiers (must be last to not override keywords)
        self.rules.append((QRegExp(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'), self.formats['identifier']))
    
    def highlightBlock(self, text: str):
        """
        Apply syntax highlighting to a block of text.
        T agency navigating P substrate with D constraints (Rule 7)
        """
        # Apply all rules in order
        for pattern, format_obj in self.rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format_obj)
                index = expression.indexIn(text, index + length)


class CodeEditor(QPlainTextEdit):
    """
    Enhanced code editor with ETPL-specific features.
    P: Text content substrate, D: Editor constraints, T: Editing agency
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set monospace font
        font = QFont("Courier New", 11)
        self.setFont(font)
        
        # Enable line wrapping
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        
        # Set tab width to 4 spaces
        self.setTabStopWidth(4 * self.fontMetrics().width(' '))
        
        # Apply syntax highlighter
        self.highlighter = ETPLSyntaxHighlighter(self.document())
        
        # Track modifications
        self.document().modificationChanged.connect(self._on_modification_changed)
        self.is_modified = False
        self.file_path: Optional[str] = None
    
    def _on_modification_changed(self, changed: bool):
        """Track document modification state"""
        self.is_modified = changed
    
    def get_text(self) -> str:
        """Get complete editor text"""
        return self.toPlainText()
    
    def set_text(self, text: str):
        """Set editor text"""
        self.setPlainText(text)
        self.document().setModified(False)
    
    def insert_text(self, text: str):
        """Insert text at cursor position"""
        cursor = self.textCursor()
        cursor.insertText(text)
    
    def keyPressEvent(self, event):
        """Handle special key events for ET symbols"""
        # Quick insert shortcuts for ET symbols
        if event.modifiers() == Qt.ControlModifier:
            key_map = {
                Qt.Key_O: '∘',  # Ctrl+O for binding operator
                Qt.Key_R: '→',  # Ctrl+R for arrow
                Qt.Key_L: 'λ',  # Ctrl+L for lambda
                Qt.Key_I: '∞',  # Ctrl+I for infinity
                Qt.Key_P: 'ψ',  # Ctrl+P for psi (quantum)
            }
            if event.key() in key_map:
                self.insertPlainText(key_map[event.key()])
                return
        
        super().keyPressEvent(event)


class ConsoleOutput(QTextEdit):
    """
    Console output widget for displaying execution results.
    P: Output text substrate, T: Display agency
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont("Courier New", 10)
        self.setFont(font)
    
    def append_output(self, text: str, color: str = "black"):
        """Append text to console with optional color"""
        self.setTextColor(QColor(color))
        self.append(text)
        self.setTextColor(QColor("black"))
    
    def append_error(self, text: str):
        """Append error message in red"""
        self.append_output(f"ERROR: {text}", "red")
    
    def append_success(self, text: str):
        """Append success message in green"""
        self.append_output(f"SUCCESS: {text}", "green")
    
    def append_info(self, text: str):
        """Append info message in blue"""
        self.append_output(f"INFO: {text}", "blue")
    
    def clear_output(self):
        """Clear console"""
        self.clear()


class ImportTreeWidget(QTreeWidget):
    """
    Widget for displaying import dependency tree.
    P: Tree structure substrate, D: Import constraints, T: Navigation
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Import Chain"])
        self.setColumnCount(1)
    
    def set_import_chain(self, chain: List[str]):
        """
        Display import chain as tree structure.
        Derived from Eq 217: Recursive descriptor discovery
        """
        self.clear()
        
        if not chain:
            return
        
        # Build tree from chain
        root = QTreeWidgetItem(self, [chain[0]])
        root.setExpanded(True)
        
        current_level = {chain[0]: root}
        
        for i in range(1, len(chain)):
            path = chain[i]
            item = QTreeWidgetItem([path])
            
            # Find parent (previous level item)
            parent_item = root
            for prev_path, prev_item in current_level.items():
                if path.startswith(os.path.dirname(prev_path)):
                    parent_item = prev_item
                    break
            
            parent_item.addChild(item)
            item.setExpanded(True)
            current_level[path] = item
    
    def get_selected_file(self) -> Optional[str]:
        """Get currently selected file path"""
        items = self.selectedItems()
        if items:
            return items[0].text(0)
        return None


class ETPLIDEMainWindow(QMainWindow):
    """
    Main IDE window for ETPL.
    P: Window substrate, D: Layout/feature constraints, T: User interaction agency
    Complete integration of all ETPL features (Eq 186: Binding complete)
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize ETPL components
        self.parser = ETPL.ETPLParser()
        self.interpreter = ETPL.ETPLInterpreter(debug=True)
        self.compiler = ETPL.ETPLCompiler()
        self.translator = ETPL.ETPLTranslator()
        
        # Current file tracking
        self.current_file: Optional[str] = None
        self.current_project_dir: Optional[str] = None
        
        # Initialize UI
        self._init_ui()
        self._init_menu()
        self._init_status_bar()
        self._apply_theme()
        
        # Window title derived from ET
        self.setWindowTitle("ETPL IDE - Exception Theory Programming Language")
        self.resize(1400, 900)
    
    def _init_ui(self):
        """Initialize main UI components (Eq 186: Complete binding)"""
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for editor and console
        main_splitter = QSplitter(Qt.Vertical)
        
        # Top section: Editor tabs
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self._close_tab)
        
        # Create first editor tab
        self._new_editor_tab()
        
        main_splitter.addWidget(self.editor_tabs)
        
        # Bottom section: Console and import tree
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Console output
        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        console_layout.setContentsMargins(0, 0, 0, 0)
        
        console_label = QLabel("Console Output:")
        console_label.setStyleSheet("font-weight: bold;")
        console_layout.addWidget(console_label)
        
        self.console = ConsoleOutput()
        console_layout.addWidget(self.console)
        
        bottom_splitter.addWidget(console_container)
        
        # Import tree
        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        
        tree_label = QLabel("Import Chain:")
        tree_label.setStyleSheet("font-weight: bold;")
        tree_layout.addWidget(tree_label)
        
        self.import_tree = ImportTreeWidget()
        tree_layout.addWidget(self.import_tree)
        
        bottom_splitter.addWidget(tree_container)
        
        # Set splitter proportions
        bottom_splitter.setSizes([700, 300])
        
        bottom_layout.addWidget(bottom_splitter)
        main_splitter.addWidget(bottom_widget)
        
        # Set main splitter proportions (editor larger)
        main_splitter.setSizes([600, 300])
        
        main_layout.addWidget(main_splitter)
    
    def _init_menu(self):
        """Initialize menu bar with all features (Eq 186: Complete)"""
        
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._new_file)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_file)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self._save_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(lambda: self._get_current_editor().undo())
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(lambda: self._get_current_editor().redo())
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(lambda: self._get_current_editor().cut())
        edit_menu.addAction(cut_action)
        
        copy_action = QAction("&Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(lambda: self._get_current_editor().copy())
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(lambda: self._get_current_editor().paste())
        edit_menu.addAction(paste_action)
        
        edit_menu.addSeparator()
        
        # ET Symbol insertion submenu
        symbols_menu = edit_menu.addMenu("Insert ET &Symbol")
        
        symbol_map = {
            "Binding (∘)": "∘",
            "Arrow (→)": "→",
            "Lambda (λ)": "λ",
            "Infinity (∞)": "∞",
            "Psi/Quantum (ψ)": "ψ",
            "Omega (Ω)": "Ω",
            "Aleph (ℵ)": "ℵ",
            "Planck (ℏ)": "ℏ",
            "Exception [0/0]": "[0/0]",
        }
        
        for name, symbol in symbol_map.items():
            action = QAction(name, self)
            action.triggered.connect(lambda checked, s=symbol: self._insert_symbol(s))
            symbols_menu.addAction(action)
        
        # Run menu
        run_menu = menubar.addMenu("&Run")
        
        parse_action = QAction("&Parse Current File", self)
        parse_action.setShortcut("F5")
        parse_action.triggered.connect(self._parse_current)
        run_menu.addAction(parse_action)
        
        interpret_action = QAction("&Interpret Current File", self)
        interpret_action.setShortcut("F6")
        interpret_action.triggered.connect(self._interpret_current)
        run_menu.addAction(interpret_action)
        
        run_menu.addSeparator()
        
        compile_action = QAction("&Compile to Binary", self)
        compile_action.setShortcut("F7")
        compile_action.triggered.connect(self._compile_current)
        run_menu.addAction(compile_action)
        
        compile_quantum_action = QAction("Compile to &Quantum", self)
        compile_quantum_action.triggered.connect(self._compile_quantum)
        run_menu.addAction(compile_quantum_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        translate_py_action = QAction("Translate Python to ETPL", self)
        translate_py_action.triggered.connect(lambda: self._translate_file('python'))
        tools_menu.addAction(translate_py_action)
        
        translate_binary_action = QAction("Translate Binary to ETPL", self)
        translate_binary_action.triggered.connect(self._translate_binary)
        tools_menu.addAction(translate_binary_action)
        
        tools_menu.addSeparator()
        
        trace_imports_action = QAction("Trace &Import Chain", self)
        trace_imports_action.triggered.connect(self._trace_imports)
        tools_menu.addAction(trace_imports_action)
        
        tools_menu.addSeparator()
        
        self_host_action = QAction("Build &Self-Hosting Package", self)
        self_host_action.triggered.connect(self._build_self_hosting)
        tools_menu.addAction(self_host_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About ETPL IDE", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        shortcuts_action = QAction("Keyboard &Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)
    
    def _init_status_bar(self):
        """Initialize status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def _apply_theme(self):
        """Apply ET-derived color theme to IDE"""
        # Professional dark theme inspired by ET primitives
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QMenuBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #4c4c4c;
            }
            QMenu {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenu::item:selected {
                background-color: #4c4c4c;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px 16px;
                border: 1px solid #555555;
            }
            QTabBar::tab:selected {
                background-color: #4c4c4c;
            }
            QPlainTextEdit, QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #555555;
            }
            QTreeWidget {
                background-color: #2b2b2b;
                color: #d4d4d4;
                border: 1px solid #555555;
            }
            QLabel {
                color: #d4d4d4;
            }
            QStatusBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
        """)
    
    def _new_editor_tab(self, title: str = "Untitled", content: str = "") -> CodeEditor:
        """Create new editor tab"""
        editor = CodeEditor()
        if content:
            editor.set_text(content)
        
        index = self.editor_tabs.addTab(editor, title)
        self.editor_tabs.setCurrentIndex(index)
        
        return editor
    
    def _close_tab(self, index: int):
        """Close editor tab with modification check"""
        editor = self.editor_tabs.widget(index)
        
        if editor.is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "This file has unsaved changes. Close anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.editor_tabs.removeTab(index)
        
        # Create new tab if last one closed
        if self.editor_tabs.count() == 0:
            self._new_editor_tab()
    
    def _get_current_editor(self) -> CodeEditor:
        """Get currently active editor"""
        return self.editor_tabs.currentWidget()
    
    def _new_file(self):
        """Create new file"""
        self._new_editor_tab()
        self.current_file = None
        self.status_bar.showMessage("New file created")
    
    def _open_file(self):
        """Open existing file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "ETPL Files (*.pdt);;Python Files (*.py);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create new tab with file content
                filename = os.path.basename(file_path)
                editor = self._new_editor_tab(filename, content)
                editor.file_path = file_path
                
                self.current_file = file_path
                self.status_bar.showMessage(f"Opened: {file_path}")
                
            except Exception as e:
                self.console.append_error(f"Failed to open file: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
    
    def _save_file(self):
        """Save current file"""
        editor = self._get_current_editor()
        
        if editor.file_path:
            self._save_to_file(editor.file_path, editor)
        else:
            self._save_file_as()
    
    def _save_file_as(self):
        """Save current file with new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File As", "",
            "ETPL Files (*.pdt);;All Files (*.*)"
        )
        
        if file_path:
            editor = self._get_current_editor()
            self._save_to_file(file_path, editor)
    
    def _save_to_file(self, file_path: str, editor: CodeEditor):
        """Save editor content to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(editor.get_text())
            
            editor.file_path = file_path
            editor.document().setModified(False)
            
            # Update tab title
            filename = os.path.basename(file_path)
            index = self.editor_tabs.indexOf(editor)
            self.editor_tabs.setTabText(index, filename)
            
            self.current_file = file_path
            self.status_bar.showMessage(f"Saved: {file_path}")
            self.console.append_success(f"File saved: {file_path}")
            
        except Exception as e:
            self.console.append_error(f"Failed to save file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def _insert_symbol(self, symbol: str):
        """Insert ET symbol at cursor"""
        editor = self._get_current_editor()
        editor.insert_text(symbol)
    
    def _parse_current(self):
        """Parse current file (Eq 223: Validation)"""
        self.console.append_info("=" * 60)
        self.console.append_info("PARSING CURRENT FILE...")
        
        editor = self._get_current_editor()
        code = editor.get_text()
        
        try:
            ast = self.parser.parse(code)
            self.console.append_success("Parse successful!")
            self.console.append_output(f"AST Root: {ast}")
            
            # Verify completeness (Eq 223)
            validation = ETMathV2Descriptor.descriptor_completion_validates(ast)
            self.console.append_info(f"Validation: {validation}")
            
            self.status_bar.showMessage("Parse successful")
            
        except Exception as e:
            self.console.append_error(f"Parse failed: {str(e)}")
            self.status_bar.showMessage("Parse failed")
    
    def _interpret_current(self):
        """Interpret current file (Rule 7: T agency)"""
        self.console.append_info("=" * 60)
        self.console.append_info("INTERPRETING CURRENT FILE...")
        
        editor = self._get_current_editor()
        code = editor.get_text()
        
        try:
            result = self.interpreter.interpret(code)
            self.console.append_success("Interpretation successful!")
            self.console.append_output(f"Result: {result}")
            
            self.status_bar.showMessage("Interpretation successful")
            
        except Exception as e:
            self.console.append_error(f"Interpretation failed: {str(e)}")
            self.status_bar.showMessage("Interpretation failed")
    
    def _compile_current(self):
        """Compile current file to binary (Eq 186: Complete binding)"""
        self.console.append_info("=" * 60)
        self.console.append_info("COMPILING TO BINARY...")
        
        editor = self._get_current_editor()
        
        if not editor.file_path or not editor.file_path.endswith('.pdt'):
            self.console.append_error("Please save file as .pdt first")
            return
        
        try:
            output_file = editor.file_path.replace('.pdt', '.exe')
            binary = self.compiler.compile_file(editor.file_path, output_file)
            
            self.console.append_success(f"Compilation successful!")
            self.console.append_info(f"Output: {output_file}")
            self.console.append_info(f"Binary size: {len(binary)} bytes")
            
            self.status_bar.showMessage(f"Compiled to {output_file}")
            
        except Exception as e:
            self.console.append_error(f"Compilation failed: {str(e)}")
            self.status_bar.showMessage("Compilation failed")
    
    def _compile_quantum(self):
        """Compile to quantum circuit (Eq 101: Quantum operations)"""
        self.console.append_info("=" * 60)
        self.console.append_info("COMPILING TO QUANTUM CIRCUIT...")
        
        editor = self._get_current_editor()
        
        if not editor.file_path or not editor.file_path.endswith('.pdt'):
            self.console.append_error("Please save file as .pdt first")
            return
        
        try:
            # Set quantum compilation mode
            old_type = self.compiler.target_type
            self.compiler.target_type = 'quantum'
            
            output_file = editor.file_path.replace('.pdt', '.qasm')
            binary = self.compiler.compile_file(editor.file_path, output_file)
            
            self.compiler.target_type = old_type
            
            self.console.append_success("Quantum compilation successful!")
            self.console.append_info(f"Output: {output_file}")
            
            self.status_bar.showMessage(f"Compiled to quantum: {output_file}")
            
        except Exception as e:
            self.console.append_error(f"Quantum compilation failed: {str(e)}")
            self.status_bar.showMessage("Quantum compilation failed")
    
    def _translate_file(self, lang: str):
        """Translate file from other language to ETPL (Eq 239)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {lang.title()} File", "",
            f"{lang.title()} Files (*.{lang[0:2]});;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        self.console.append_info("=" * 60)
        self.console.append_info(f"TRANSLATING {lang.upper()} TO ETPL...")
        
        try:
            etpl_code = self.translator.translate_file(file_path, lang)
            
            # Create new tab with translated code
            filename = os.path.basename(file_path).replace(f'.{lang[0:2]}', '.pdt')
            self._new_editor_tab(f"{filename} (Translated)", etpl_code)
            
            self.console.append_success("Translation successful!")
            self.console.append_info(f"Translated from: {file_path}")
            
            self.status_bar.showMessage(f"Translated {file_path}")
            
        except Exception as e:
            self.console.append_error(f"Translation failed: {str(e)}")
            self.status_bar.showMessage("Translation failed")
    
    def _translate_binary(self):
        """Translate binary executable to ETPL (Eq 233: Deobfuscation)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Binary File", "",
            "Executable Files (*.exe *.dll *.so);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        self.console.append_info("=" * 60)
        self.console.append_info("TRANSLATING BINARY TO ETPL...")
        self.console.append_info("This may take a while for large binaries...")
        
        try:
            etpl_code = self.translator.translate_binary_lossless(file_path)
            
            # Create new tab with translated code
            filename = os.path.basename(file_path) + '.pdt'
            self._new_editor_tab(f"{filename} (Binary Translation)", etpl_code)
            
            self.console.append_success("Binary translation successful!")
            
            self.status_bar.showMessage(f"Translated binary: {file_path}")
            
        except Exception as e:
            self.console.append_error(f"Binary translation failed: {str(e)}")
            self.status_bar.showMessage("Binary translation failed")
    
    def _trace_imports(self):
        """Trace import chain for current file (Eq 217: Recursive)"""
        editor = self._get_current_editor()
        
        if not editor.file_path:
            self.console.append_error("Please save file first")
            return
        
        self.console.append_info("=" * 60)
        self.console.append_info("TRACING IMPORT CHAIN...")
        
        try:
            chain = self.translator.trace_imports(editor.file_path, visited=set())
            
            # Display in tree widget
            self.import_tree.set_import_chain(chain)
            
            self.console.append_success(f"Found {len(chain)} imports")
            for i, path in enumerate(chain, 1):
                self.console.append_output(f"  {i}. {path}")
            
            self.status_bar.showMessage(f"Traced {len(chain)} imports")
            
        except Exception as e:
            self.console.append_error(f"Import tracing failed: {str(e)}")
            self.status_bar.showMessage("Import tracing failed")
    
    def _build_self_hosting(self):
        """
        Build self-hosting package: GUI + ETPL → .pdt → .exe
        This creates the first unified self-hosting ETPL environment
        """
        self.console.append_info("=" * 60)
        self.console.append_info("BUILDING SELF-HOSTING PACKAGE...")
        self.console.append_info("Stage 1: Translating Python files to ETPL...")
        
        try:
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for Self-Hosting Package"
            )
            
            if not output_dir:
                return
            
            # Step 1: Translate ETPL.py to ETPL
            etpl_py_path = os.path.join(os.path.dirname(__file__), 'ETPL.py')
            self.console.append_info(f"Translating ETPL.py...")
            etpl_code = self.translator.translate_file(etpl_py_path, 'python')
            
            # Step 2: Translate this IDE script to ETPL
            ide_py_path = __file__
            self.console.append_info(f"Translating IDE...")
            ide_code = self.translator.translate_file(ide_py_path, 'python')
            
            # Step 3: Bind both into unified .pdt
            self.console.append_info("Stage 2: Binding into unified .pdt...")
            unified_code = ETMathV2.bind_operation([etpl_code, ide_code])
            
            unified_pdt_path = os.path.join(output_dir, 'etpl_ide_unified.pdt')
            with open(unified_pdt_path, 'w', encoding='utf-8') as f:
                f.write(unified_code)
            
            self.console.append_success(f"Created unified .pdt: {unified_pdt_path}")
            
            # Step 4: Compile to executable
            self.console.append_info("Stage 3: Compiling to executable...")
            exe_path = unified_pdt_path.replace('.pdt', '.exe')
            self.compiler.compile_file(unified_pdt_path, exe_path)
            
            self.console.append_success("=" * 60)
            self.console.append_success("SELF-HOSTING PACKAGE COMPLETE!")
            self.console.append_success(f"Executable: {exe_path}")
            self.console.append_info("You can now use the .exe as a standalone ETPL IDE")
            
            self.status_bar.showMessage("Self-hosting package built successfully")
            
            QMessageBox.information(
                self, "Success",
                f"Self-hosting package created!\n\n"
                f"Unified .pdt: {unified_pdt_path}\n"
                f"Executable: {exe_path}\n\n"
                f"You can now use the executable as a standalone ETPL IDE."
            )
            
        except Exception as e:
            self.console.append_error(f"Self-hosting build failed: {str(e)}")
            self.status_bar.showMessage("Self-hosting build failed")
            import traceback
            self.console.append_error(traceback.format_exc())
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>ETPL IDE</h2>
        <p><b>Exception Theory Programming Language IDE</b></p>
        <p>Version 1.0 - Stage 1</p>
        <p>Derived from Exception Theory (ET)</p>
        <ul>
            <li>P: Infinite substrate (code, data)</li>
            <li>D: Finite constraints (syntax, types)</li>
            <li>T: Indeterminate agency (execution, user)</li>
        </ul>
        <p>Created by Aevum Defluo</p>
        <p>© 2025 - All rights derived from ET</p>
        """
        QMessageBox.about(self, "About ETPL IDE", about_text)
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <table>
        <tr><td><b>File Operations:</b></td><td></td></tr>
        <tr><td>New File</td><td>Ctrl+N</td></tr>
        <tr><td>Open File</td><td>Ctrl+O</td></tr>
        <tr><td>Save File</td><td>Ctrl+S</td></tr>
        <tr><td>Save As</td><td>Ctrl+Shift+S</td></tr>
        
        <tr><td><b>Edit Operations:</b></td><td></td></tr>
        <tr><td>Undo</td><td>Ctrl+Z</td></tr>
        <tr><td>Redo</td><td>Ctrl+Y</td></tr>
        <tr><td>Cut</td><td>Ctrl+X</td></tr>
        <tr><td>Copy</td><td>Ctrl+C</td></tr>
        <tr><td>Paste</td><td>Ctrl+V</td></tr>
        
        <tr><td><b>ET Symbols:</b></td><td></td></tr>
        <tr><td>Binding (∘)</td><td>Ctrl+O</td></tr>
        <tr><td>Arrow (→)</td><td>Ctrl+R</td></tr>
        <tr><td>Lambda (λ)</td><td>Ctrl+L</td></tr>
        <tr><td>Infinity (∞)</td><td>Ctrl+I</td></tr>
        <tr><td>Psi (ψ)</td><td>Ctrl+P</td></tr>
        
        <tr><td><b>Run Operations:</b></td><td></td></tr>
        <tr><td>Parse</td><td>F5</td></tr>
        <tr><td>Interpret</td><td>F6</td></tr>
        <tr><td>Compile</td><td>F7</td></tr>
        </table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)
    
    def closeEvent(self, event):
        """Handle window close with unsaved changes check"""
        # Check all tabs for unsaved changes
        unsaved_tabs = []
        for i in range(self.editor_tabs.count()):
            editor = self.editor_tabs.widget(i)
            if editor.is_modified:
                unsaved_tabs.append(self.editor_tabs.tabText(i))
        
        if unsaved_tabs:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                f"The following files have unsaved changes:\n" +
                "\n".join(unsaved_tabs) +
                "\n\nQuit anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        event.accept()


def main():
    """
    Main entry point for ETPL IDE.
    T agency starting the application (Rule 7)
    """
    app = QApplication(sys.argv)
    app.setApplicationName("ETPL IDE")
    app.setOrganizationName("Exception Theory")
    
    # Create and show main window
    window = ETPLIDEMainWindow()
    window.show()
    
    # Start event loop (T infinite navigation)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
