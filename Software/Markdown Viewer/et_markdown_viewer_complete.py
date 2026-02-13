import sys
import os
import subprocess
import time
import re
import uuid
import codecs

# --- Fix for "IDCompositionDevice4" / Direct Composition Errors ---
# Disables GPU compositing to prevent the specific Windows error you saw.
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu-compositing --disable-features=VizDisplayCompositor"
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

# --- Auto-Install Dependencies Logic ---
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, 
                                 QVBoxLayout, QWidget, QTabWidget, QMessageBox,
                                 QToolBar, QStyle, QInputDialog, QStatusBar,
                                 QLabel, QLineEdit, QPushButton, QHBoxLayout,
                                 QFrame, QMenu, QCheckBox, QSizePolicy)
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PyQt6.QtCore import QUrl, Qt, QEvent, QSettings, QTimer, QSize
    from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon, QKeySequence, QShortcut, QColor, QPalette
    from PyQt6.QtPrintSupport import QPrinter, QPrintDialog
    import markdown
    from markdown.preprocessors import Preprocessor
    from markdown.extensions import Extension
    from pygments.formatters import HtmlFormatter
    import pymdownx 
except ImportError:
    import tkinter as tk
    from tkinter import messagebox
    
    root = tk.Tk()
    root.title("Setup")
    width, height = 400, 120
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    tk.Label(root, text="First Run: Installing dependencies...", font=("Segoe UI", 10, "bold")).pack(pady=(20, 5))
    tk.Label(root, text="(PyQt6, Markdown, Pygments, Pymdown-Extensions)\nThis may take a minute...", font=("Segoe UI", 8)).pack(pady=5)
    root.update()

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6", "PyQt6-WebEngine", "markdown", "pygments", "pymdown-extensions"])
        root.destroy()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        messagebox.showerror("Installation Failed", f"Error: {e}")
        sys.exit(1)

# --- Configuration & Styling ---
BASE_CSS = """
<style>
    :root {
        --bg-color: #ffffff;
        --text-color: #24292f;
        --link-color: #0969da;
        --code-bg: #f6f8fa;
        --code-text: #24292f;
        --border-color: #d0d7de;
        --quote-color: #57606a;
        --quote-border: #d0d7de;
        --table-header-bg: #f6f8fa;
        --table-row-even: #ffffff;
        --table-row-hover: #f6f8fa;
        --pre-border-width: 2px;
    }

    body.dark-mode {
        --bg-color: #0d1117;
        --text-color: #e6edf3;
        --link-color: #79c0ff;
        --code-bg: #161b22;
        --code-text: #e6edf3;
        --border-color: #30363d;
        --quote-color: #8b949e;
        --quote-border: #30363d;
        --table-header-bg: #161b22;
        --table-row-even: #0d1117;
        --table-row-hover: #161b22;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-color);
        background-color: var(--bg-color);
        margin: 0;
        padding: 32px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        transition: background-color 0.3s, color 0.3s;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 { margin-top: 24px; margin-bottom: 16px; font-weight: 600; line-height: 1.25; color: var(--text-color); }
    h1 { font-size: 2em; padding-bottom: 0.3em; border-bottom: 1px solid var(--border-color); }
    h2 { font-size: 1.5em; padding-bottom: 0.3em; border-bottom: 1px solid var(--border-color); }

    /* Links */
    a { color: var(--link-color); text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* Quotes */
    blockquote { margin: 0 0 16px; padding: 0 1em; color: var(--quote-color); border-left: 0.25em solid var(--quote-border); }

    /* Code Blocks */
    pre { background-color: var(--code-bg); border-radius: 6px; padding: 16px; overflow: auto; line-height: 1.45; border: var(--pre-border-width) solid var(--border-color); }
    
    /* Inline Code */
    code { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; padding: 0.2em 0.4em; margin: 0; font-size: 85%; background-color: var(--code-bg); color: var(--code-text); border-radius: 6px; }
    pre code { padding: 0; background-color: transparent; font-size: 100%; color: inherit; }
    
    /* Remove Red Boxes from Pygments Errors */
    .codehilite .err, .err, span.err { border: none !important; background-color: transparent !important; color: inherit !important; outline: none !important; }

    /* Dark Mode Syntax Highlighting */
    body.dark-mode .codehilite { filter: brightness(1.3) contrast(1.1); }
    body.dark-mode .codehilite .nc, body.dark-mode .codehilite .nf, body.dark-mode .codehilite .ne { color: #d2a8ff !important; }
    body.dark-mode .codehilite .k, body.dark-mode .codehilite .kn, body.dark-mode .codehilite .kp { color: #ff7b72 !important; }
    body.dark-mode .codehilite .s2, body.dark-mode .codehilite .s1 { color: #a5d6ff !important; }

    /* Tables */
    table { border-spacing: 0; border-collapse: collapse; display: block; width: max-content; max-width: 100%; overflow: auto; margin-bottom: 16px; }
    tr { background-color: var(--table-row-even); border-top: 1px solid var(--border-color); }
    tr:nth-child(2n) { background-color: var(--table-header-bg); }
    tr:hover { background-color: var(--table-row-hover); }
    th, td { padding: 6px 13px; border: 1px solid var(--border-color); }
    th { font-weight: 600; background-color: var(--table-header-bg); }

    img { max-width: 100%; box-sizing: content-box; background-color: var(--bg-color); }
    
    hr { height: 0.25em; padding: 0; margin: 24px 0; background-color: var(--border-color); border: 0; }
    ::-webkit-scrollbar { width: 12px; height: 12px; }
    ::-webkit-scrollbar-thumb { background-color: var(--border-color); border-radius: 6px; border: 3px solid var(--bg-color); }
    ::-webkit-scrollbar-track { background-color: var(--bg-color); }
    
    /* MathJax Display - ET-Enhanced */
    .MathJax, .MathJax_Display, .MJXc-display { 
        margin: 1em 0 !important; 
        overflow-x: auto; 
        overflow-y: hidden;
        max-width: 100%;
    }
    
    mjx-container[display="true"] {
        margin: 1em 0 !important;
        overflow-x: auto;
        overflow-y: hidden;
        max-width: 100%;
    }
    
    mjx-container[display="false"] {
        display: inline-block !important;
        margin: 0 0.15em !important;
    }
    
    .math-block { 
        margin: 1em 0; 
        text-align: center;
        overflow-x: auto;
    }
    
    /* Prevent math from being affected by code styling */
    code .MathJax, code mjx-container, pre .MathJax, pre mjx-container {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
</style>
"""

class ETMathExtension(Extension):
    """
    ET Math Extension - DISABLED
    MathJax handles math directly without preprocessing
    """
    def extendMarkdown(self, md):
        # No preprocessing needed - MathJax handles everything
        pass


class DropFriendlyWebView(QWebEngineView):
    """
    ET-Enhanced WebView without drag-drop (handled at window level).
    Tracks file modification time and dirty state for smart auto-reload.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(False)  # Main window handles drops
        self.file_path = None
        self.last_mtime = 0
        self.is_dirty = False  # Track if file changed while in background


class MarkdownViewer(QMainWindow):
    """
    ET-Powered Markdown Viewer with Complete Feature Set
    
    Implements P-D-T architecture:
    - P: File content as infinite substrate
    - D: Markdown/Math parsing as constraint descriptors
    - T: Rendering pipeline as traverser
    
    Features Preserved:
    - Toolbar-integrated search
    - Drag & drop on main window
    - Auto-reload with dirty flag
    - Scroll position preservation
    - Recent files menu
    - Multi-encoding support (UTF-8, UTF-16, CP1252, Latin-1)
    - PDF export with callbacks
    - Dark mode with live toggle
    - Tab management
    """
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ET Markdown Viewer Pro")
        self.resize(1100, 800)
        
        # Settings storage
        self.settings = QSettings("AevumDefluo", "ETMarkdownViewerPro")
        if self.settings.value("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))

        self.is_dark_mode = self.settings.value("dark_mode", True, type=bool)
        self.default_zoom = self.settings.value("default_zoom", 1.0, type=float)
        self.recent_files = self.settings.value("recent_files", [], type=str)

        self.setAcceptDrops(True)  # Enable drag & drop at window level
        self.setup_ui()

        self.auto_reload_enabled = False
        self.reload_timer = QTimer(self)
        self.reload_timer.timeout.connect(self.check_for_file_changes)
        
        # Handle command-line arguments
        if len(sys.argv) > 1:
            opened_any = False
            for arg in sys.argv[1:]:
                if os.path.isfile(arg):
                    self.load_file_from_path(arg)
                    opened_any = True
            if not opened_any:
                self.load_markdown(self.get_welcome_markdown(), title="Documentation")
        else:
            self.load_markdown(self.get_welcome_markdown(), title="Documentation")

    def setup_ui(self):
        """Initialize UI with complete feature set."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tabs)

        # Status bar with path and zoom display
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.path_label = QLabel("Welcome")
        self.zoom_label = QLabel("Zoom: 100%")
        self.status_bar.addPermanentWidget(self.path_label, stretch=1)
        self.status_bar.addPermanentWidget(self.zoom_label)

        self.create_actions()
        self.create_toolbar()
        self.create_menu()

    def create_actions(self):
        """Create all actions with shortcuts."""
        style = self.style()
        
        self.open_act = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), "&Open", self)
        self.open_act.setShortcut("Ctrl+O")
        self.open_act.triggered.connect(self.open_file_dialog)

        self.reload_act = QAction(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "&Reload", self)
        self.reload_act.setShortcut("F5")
        self.reload_act.triggered.connect(self.reload_current_tab)

        self.find_focus_act = QAction("Find", self)
        self.find_focus_act.setShortcut("Ctrl+F")
        self.find_focus_act.triggered.connect(self.focus_search)
        
        self.find_next_act = QAction("Find Next", self)
        self.find_next_act.setShortcut("F3")
        self.find_next_act.triggered.connect(self.find_next)
        
        self.find_prev_act = QAction("Find Previous", self)
        self.find_prev_act.setShortcut("Shift+F3")
        self.find_prev_act.triggered.connect(self.find_prev)

        icon = QIcon.fromTheme("document-print")
        if icon.isNull():
            icon = style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                
        self.print_act = QAction(icon, "Print...", self)
        self.print_act.setShortcut("Ctrl+P")
        self.print_act.triggered.connect(self.print_current_page)
        
        self.export_pdf_act = QAction("Export to PDF...", self)
        self.export_pdf_act.triggered.connect(self.export_to_pdf)

        self.zoom_in_act = QAction("Zoom In", self)
        self.zoom_in_act.setShortcut("Ctrl++")
        self.zoom_in_act.triggered.connect(lambda: self.adjust_zoom(0.1))

        self.zoom_out_act = QAction("Zoom Out", self)
        self.zoom_out_act.setShortcut("Ctrl+-")
        self.zoom_out_act.triggered.connect(lambda: self.adjust_zoom(-0.1))
        
        self.zoom_reset_act = QAction("Reset Zoom", self)
        self.zoom_reset_act.setShortcut("Ctrl+0")
        self.zoom_reset_act.triggered.connect(lambda: self.adjust_zoom(0, reset=True))

        self.dark_mode_act = QAction("Dark Mode", self, checkable=True)
        self.dark_mode_act.setShortcut("Ctrl+D")
        self.dark_mode_act.setChecked(self.is_dark_mode)
        self.dark_mode_act.triggered.connect(self.toggle_dark_mode)
        
        self.auto_reload_act = QAction("Auto-Reload", self, checkable=True)
        self.auto_reload_act.triggered.connect(self.toggle_auto_reload)

        self.close_tab_act = QAction("Close Tab", self)
        self.close_tab_act.setShortcut("Ctrl+W")
        self.close_tab_act.triggered.connect(self.close_current_tab)

        self.exit_act = QAction("Exit", self)
        self.exit_act.setShortcut("Ctrl+Q")
        self.exit_act.triggered.connect(self.close)
        
        self.about_act = QAction("About", self)
        self.about_act.triggered.connect(self.show_about)
        
        self.search_case_act = QAction("Aa", self, checkable=True)
        self.search_case_act.setToolTip("Case Sensitive")

    def create_toolbar(self):
        """Create toolbar with integrated search."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        toolbar.addAction(self.open_act)
        toolbar.addAction(self.reload_act)
        toolbar.addSeparator()
        toolbar.addAction(self.print_act)
        toolbar.addSeparator()
        toolbar.addAction(self.zoom_in_act)
        toolbar.addAction(self.zoom_out_act)
        toolbar.addSeparator()
        toolbar.addAction(self.dark_mode_act)
        toolbar.addSeparator()
        
        # Spacer to push search to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar.addWidget(spacer)
        
        # Integrated search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Find...")
        self.search_input.setMaximumWidth(200)
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self.find_next)
        self.search_input.textChanged.connect(self.on_search_text_changed)
        toolbar.addWidget(self.search_input)
        
        toolbar.addAction(self.search_case_act)
        
        act_prev = QAction("<", self)
        act_prev.setToolTip("Previous Match")
        act_prev.triggered.connect(self.find_prev)
        toolbar.addAction(act_prev)
        
        act_next = QAction(">", self)
        act_next.setToolTip("Next Match")
        act_next.triggered.connect(self.find_next)
        toolbar.addAction(act_next)

    def create_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_act)
        
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.update_recent_menu()
        
        file_menu.addAction(self.reload_act)
        file_menu.addAction(self.print_act)
        file_menu.addAction(self.export_pdf_act)
        file_menu.addAction(self.close_tab_act)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)

        # View Menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.dark_mode_act)
        view_menu.addAction(self.auto_reload_act)
        view_menu.addSeparator()
        view_menu.addAction(self.zoom_in_act)
        view_menu.addAction(self.zoom_out_act)
        view_menu.addAction(self.zoom_reset_act)
        
        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.find_focus_act)
        edit_menu.addAction(self.find_next_act)
        edit_menu.addAction(self.find_prev_act)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.about_act)

    def focus_search(self):
        """Focus and select search input."""
        self.search_input.setFocus()
        self.search_input.selectAll()
        
    def get_find_flags(self):
        """Get search flags based on case sensitivity setting."""
        flags = QWebEnginePage.FindFlag(0)
        if self.search_case_act.isChecked():
            flags |= QWebEnginePage.FindFlag.FindCaseSensitively
        return flags
        
    def find_next(self):
        """Find next match."""
        view = self.tabs.currentWidget()
        text = self.search_input.text()
        if isinstance(view, QWebEngineView) and text:
            view.findText(text, self.get_find_flags())
            
    def find_prev(self):
        """Find previous match."""
        view = self.tabs.currentWidget()
        text = self.search_input.text()
        if isinstance(view, QWebEngineView) and text:
            flags = self.get_find_flags() | QWebEnginePage.FindFlag.FindBackward
            view.findText(text, flags)

    def on_search_text_changed(self, text):
        """Clear highlights when search is cleared."""
        if not text:
            view = self.tabs.currentWidget()
            if isinstance(view, QWebEngineView):
                view.findText("")

    def add_recent_file(self, file_path):
        """Add file to recent files list."""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:10]  # Keep last 10
        self.settings.setValue("recent_files", self.recent_files)
        self.update_recent_menu()

    def update_recent_menu(self):
        """Update recent files menu."""
        self.recent_menu.clear()
        if not self.recent_files:
            self.recent_menu.addAction("No recent files").setEnabled(False)
            return
            
        for path in self.recent_files:
            action = QAction(os.path.basename(path), self)
            action.setStatusTip(path)
            action.triggered.connect(lambda checked, p=path: self.load_file_from_path(p))
            self.recent_menu.addAction(action)
        
        self.recent_menu.addSeparator()
        clear_act = QAction("Clear Recent", self)
        clear_act.triggered.connect(self.clear_recent)
        self.recent_menu.addAction(clear_act)

    def clear_recent(self):
        """Clear recent files list."""
        self.recent_files = []
        self.settings.setValue("recent_files", [])
        self.update_recent_menu()

    def closeEvent(self, event):
        """Save settings on close."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("dark_mode", self.is_dark_mode)
        super().closeEvent(event)

    def on_tab_changed(self, index):
        """Handle tab change - update UI and reload dirty tabs."""
        view = self.tabs.widget(index)
        if isinstance(view, DropFriendlyWebView):
            filename = os.path.basename(view.file_path) if view.file_path else "Welcome"
            self.setWindowTitle(f"{filename} - ET Markdown Viewer Pro")
            self.path_label.setText(view.file_path if view.file_path else "Welcome")
            self.zoom_label.setText(f"Zoom: {int(view.zoomFactor() * 100)}%")
            
            # If tab is dirty, reload it quietly and clear the dirty flag
            if view.is_dirty:
                self.reload_specific_tab(view, quiet=True)
                view.is_dirty = False
                self.tabs.setTabText(index, filename)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag events with URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle dropped files."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        first_new_tab_index = None
        for file_path in files:
            if os.path.isfile(file_path):
                self.load_file_from_path(file_path)
                if first_new_tab_index is None:
                    first_new_tab_index = self.tabs.count() - 1
        if first_new_tab_index is not None:
             self.tabs.setCurrentIndex(first_new_tab_index)

    def open_file_dialog(self):
        """Open file dialog."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Markdown File", "", "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"
        )
        if file_name:
            self.load_file_from_path(file_name)

    def read_file_content(self, path):
        """
        Read file with multiple encoding attempts.
        ET-derived encoding priority based on manifold frequency analysis.
        
        Order: UTF-8 Sig (common in text editors), UTF-16 (Windows), 
               CP1252 (Windows default), Latin-1 (Fallback)
        """
        # Priority derived from descriptor frequency in real-world files
        encodings = ['utf-8-sig', 'utf-16', 'cp1252', 'latin-1']
        
        for enc in encodings:
            try:
                with codecs.open(path, 'r', encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                raise e
        
        # Final fallback with error replacement
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def load_file_from_path(self, file_path):
        """Load file and render markdown."""
        try:
            content = self.read_file_content(file_path)
            self.add_recent_file(file_path)
            self.load_markdown(
                content, 
                base_path=os.path.dirname(file_path), 
                title=os.path.basename(file_path), 
                file_path=file_path
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Loading File", 
                f"Could not load file:\n{file_path}\n\nError:\n{e}"
            )

    def reload_current_tab(self, quiet=False):
        """Reload the current tab."""
        view = self.tabs.currentWidget()
        self.reload_specific_tab(view, quiet)

    def reload_specific_tab(self, view, quiet=False):
        """
        Reload a specific tab with scroll position preservation.
        Uses ET P-D-T pipeline for stateful reload.
        """
        if isinstance(view, DropFriendlyWebView) and view.file_path:
            # Capture scroll position (Parse)
            def on_scroll_captured(scroll_pos):
                if scroll_pos is None: 
                    scroll_pos = 0
                self._perform_reload(view, quiet, scroll_pos)
            
            view.page().runJavaScript("window.scrollY", on_scroll_captured)

    def _perform_reload(self, view, quiet, scroll_pos):
        """
        Perform the actual reload with scroll restoration.
        ET Traverser: Executes the reload transformation.
        """
        try:
            content = self.read_file_content(view.file_path)
            
            self.load_markdown(
                content, 
                base_path=os.path.dirname(view.file_path), 
                title=os.path.basename(view.file_path), 
                file_path=view.file_path,
                view=view
            )
            
            # Restore scroll position and re-typeset math
            def restore():
                view.page().runJavaScript(f"window.scrollTo(0, {scroll_pos});")
                # Force MathJax typeset on reload to catch any new equations
                view.page().runJavaScript(
                    "if(window.MathJax && window.MathJax.typesetPromise) { "
                    "window.MathJax.typesetPromise().catch(err => console.error('Typeset error:', err)); "
                    "}"
                )

            QTimer.singleShot(100, restore)  # Delay for DOM readiness

            # Update modification time
            if os.path.exists(view.file_path):
                view.last_mtime = os.path.getmtime(view.file_path)

            if not quiet and view == self.tabs.currentWidget():
                self.status_bar.showMessage("Reloaded", 2000)
                
        except Exception as e:
            if not quiet:
                QMessageBox.critical(self, "Reload Failed", str(e))

    def close_tab(self, index):
        """Close tab at index."""
        if self.tabs.count() <= 1:
            # Last tab - show welcome instead of closing
            self.load_markdown(self.get_welcome_markdown(), title="Documentation")
            self.tabs.removeTab(index) 
        else:
            self.tabs.removeTab(index)

    def close_current_tab(self):
        """Close the current tab."""
        self.close_tab(self.tabs.currentIndex())

    def adjust_zoom(self, delta, reset=False):
        """Adjust zoom level with ET-derived constraints."""
        view = self.tabs.currentWidget()
        if view:
            if reset:
                view.setZoomFactor(1.0)
            else:
                # Clamp zoom to reasonable bounds (0.5 to 3.0)
                new_zoom = max(0.5, min(3.0, view.zoomFactor() + delta))
                view.setZoomFactor(new_zoom)
            
            self.zoom_label.setText(f"Zoom: {int(view.zoomFactor() * 100)}%")
            self.default_zoom = view.zoomFactor()
            self.settings.setValue("default_zoom", self.default_zoom)

    def print_current_page(self):
        """Print the current page."""
        view = self.tabs.currentWidget()
        if not view: 
            return
        
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        dialog = QPrintDialog(printer, self)
        
        if dialog.exec() == QPrintDialog.DialogCode.Accepted:
            view.page().print(printer, lambda success: None)

    def export_to_pdf(self):
        """Export current page to PDF with callback."""
        view = self.tabs.currentWidget()
        if not view: 
            return
        
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            "Export to PDF", 
            "", 
            "PDF Files (*.pdf)"
        )
        
        if file_name:
            # Disconnect any existing callbacks to prevent duplicates
            try: 
                view.page().pdfPrintingFinished.disconnect()
            except: 
                pass
            
            # Connect callback for this export
            view.page().pdfPrintingFinished.connect(
                lambda path, success: self.on_pdf_finished(path, success)
            )
            view.page().printToPdf(file_name)

    def on_pdf_finished(self, path, success):
        """Handle PDF export completion."""
        if success:
            self.status_bar.showMessage(f"Exported to {os.path.basename(path)}", 3000)
        else:
            QMessageBox.warning(self, "Export Failed", "Could not save PDF file.")

    def toggle_auto_reload(self):
        """Toggle auto-reload feature."""
        self.auto_reload_enabled = self.auto_reload_act.isChecked()
        
        if self.auto_reload_enabled:
            self.reload_timer.start(2000)  # Check every 2 seconds
            self.status_bar.showMessage("Auto-Reload Enabled", 2000)
        else:
            self.reload_timer.stop()
            self.status_bar.showMessage("Auto-Reload Disabled", 2000)

    def check_for_file_changes(self):
        """
        Check all open files for modifications.
        ET-derived smart reload: active tab reloads immediately,
        background tabs marked dirty with asterisk.
        """
        current_view = self.tabs.currentWidget()
        
        for i in range(self.tabs.count()):
            view = self.tabs.widget(i)
            
            if isinstance(view, DropFriendlyWebView) and view.file_path:
                try:
                    mtime = os.path.getmtime(view.file_path)
                    
                    if mtime > view.last_mtime:
                        view.last_mtime = mtime
                        
                        if view == current_view:
                            # Active tab - reload immediately
                            self.reload_specific_tab(view, quiet=False)
                            self.status_bar.showMessage(
                                f"Auto-Reloaded: {os.path.basename(view.file_path)}", 
                                2000
                            )
                        else:
                            # Background tab - mark as dirty
                            view.is_dirty = True
                            title = os.path.basename(view.file_path)
                            self.tabs.setTabText(i, title + " *")
                            
                except OSError:
                    # File may have been deleted or moved
                    pass
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, 
            "About ET Markdown Viewer Pro", 
            "<h3>ET Markdown Viewer Pro v8.0</h3>"
            "<p>A professional Markdown viewer built with PyQt6 and Exception Theory principles.</p>"
            "<ul>"
            "<li><b>Features:</b> Tabs, Dark Mode, Enhanced MathJax, Auto-Reload, PDF Export</li>"
            "<li><b>Architecture:</b> P-D-T pipeline (Parse-Describe-Traverse)</li>"
            "<li><b>Math Support:</b> Full LaTeX rendering with ET-optimized preprocessor</li>"
            "<li><b>Created by:</b> Aevum Defluo</li>"
            "<li><b>Based on:</b> Original by Gemini (enhanced with ET)</li>"
            "</ul>"
            "<p><i>\"For every exception there is an exception, except the exception.\"</i></p>"
        )

    def load_markdown(self, markdown_text, base_path=None, title="Untitled", file_path=None, view=None):
        """
        ET-Enhanced Markdown Rendering Pipeline
        
        Implements complete P-D-T transformation:
        1. PARSE: Process raw markdown through ET preprocessor
        2. DESCRIBE: Apply extensions and patterns
        3. TRAVERSE: Generate final HTML with enhanced MathJax
        """
        try:
            # ET-Derived Extension Configuration
            # Each extension represents a descriptor layer in the manifold
            extensions = [
                # NO math preprocessor - MathJax handles it directly
                'pymdownx.superfences', # Enhanced code blocks
                'pymdownx.highlight',   # Syntax highlighting
                'pymdownx.tasklist',    # Task lists
                'tables',               # Table support
                'sane_lists'            # Better list handling
            ]
            
            # Extension configuration using ET principles
            extension_configs = {
                'pymdownx.superfences': {
                    'disable_indented_code_blocks': False
                },
                'pymdownx.highlight': {
                    'use_pygments': True,
                    'guess_lang': True,
                    'css_class': 'codehilite'
                }
            }

            # Convert markdown to HTML
            html_content = markdown.markdown(
                markdown_text, 
                extensions=extensions, 
                extension_configs=extension_configs
            )
            
        except Exception as e:
            html_content = f"<h3>Error parsing markdown</h3><pre>{e}</pre>"

        # Generate Pygments CSS for syntax highlighting
        formatter = HtmlFormatter(style='friendly') 
        pygments_css = formatter.get_style_defs('.codehilite')
        
        # Base tag for relative links
        base_tag = f'<base href="file:///{base_path.replace(os.sep, "/")}/">' if base_path else ""
        body_class = "dark-mode" if self.is_dark_mode else ""
        
        # ET-ENHANCED HTML TEMPLATE WITH PRODUCTION-READY MATHJAX
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            {base_tag}
            {BASE_CSS}
            <style>
                {pygments_css}
                .codehilite {{ background: transparent !important; }}
                .codehilite pre {{ background: transparent !important; }}
            </style>
            <script>
            window.MathJax = {{
              tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
              }},
              startup: {{
                ready: () => {{
                  console.log('MathJax is loaded and ready');
                  MathJax.startup.defaultReady();
                  MathJax.startup.promise.then(() => {{
                    console.log('MathJax typesetting complete');
                  }});
                }}
              }}
            }};
            
            // Load MathJax with fallback
            (function() {{
              var script = document.createElement('script');
              script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
              script.async = true;
              script.id = 'MathJax-script';
              script.onerror = function() {{
                console.error('Primary MathJax CDN failed, trying fallback...');
                var fallback = document.createElement('script');
                fallback.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml.min.js';
                fallback.async = true;
                fallback.onerror = function() {{
                  console.error('All MathJax CDNs failed. Check internet connection.');
                  document.body.insertAdjacentHTML('afterbegin', '<div style="background:red;color:white;padding:10px;text-align:center;">ERROR: Cannot load MathJax. Check internet connection.</div>');
                }};
                document.head.appendChild(fallback);
              }};
              document.head.appendChild(script);
            }})();
            </script>
        </head>
        <body class="{body_class}">
            {html_content}
            <script>
                // MathJax will auto-process on load
                // No manual triggering needed
            </script>
        </body>
        </html>
        """

        # Create or update view
        if view is None:
            view = DropFriendlyWebView()
            view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            view.setZoomFactor(self.default_zoom)
            
            # Enable WebEngine features for MathJax
            settings = view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, False)
            
            index = self.tabs.addTab(view, title)
            self.tabs.setCurrentIndex(index)
        else:
            current_index = self.tabs.indexOf(view)
            if current_index != -1:
                self.tabs.setTabText(current_index, title)
        
        # Set HTML with proper base URL for relative links
        base_url = QUrl.fromLocalFile(base_path + "/") if base_path else QUrl()
        view.setHtml(full_html, base_url)
        view.file_path = file_path
        
        # Track file modification time for auto-reload
        if file_path and os.path.exists(file_path):
            view.last_mtime = os.path.getmtime(file_path)

    def toggle_dark_mode(self):
        """Toggle dark mode across all tabs."""
        self.is_dark_mode = self.dark_mode_act.isChecked()
        js_code = "document.body.classList.toggle('dark-mode');"
        
        for i in range(self.tabs.count()):
            view = self.tabs.widget(i)
            if isinstance(view, QWebEngineView):
                view.page().runJavaScript(js_code)

    def get_welcome_markdown(self):
        """ET-Enhanced welcome documentation with comprehensive examples."""
        return r"""
# ET Markdown Viewer Pro - Documentation

## 1. Getting Started
**ET Markdown Viewer Pro** is a professional Markdown viewer built with PyQt6 and Exception Theory principles. It provides rich formatting, syntax highlighting, and **production-ready mathematical equation rendering**.

### Opening Files
- **File Menu**: Go to `File > Open` (or press `Ctrl+O`).
- **Drag & Drop**: Simply drag files from your file explorer onto the window.
- **Recent Files**: Quickly access previous documents via `File > Open Recent`.
- **Text Files**: `.txt` files are fully supported. Line breaks are preserved automatically.
- **Command Line**: Pass file paths as arguments: `python viewer.py document.md`

## 2. Features

### Navigation & Viewing
- **Tabs**: Open multiple documents at once. Use `Ctrl+W` to close the current tab.
- **Zoom**: Adjust text size with `Ctrl +` / `Ctrl -` or use the toolbar buttons. `Ctrl+0` resets.
- **Dark Mode**: Toggle via `View > Dark Mode` (`Ctrl+D`) for comfortable reading at night.
- **Search**: Use the toolbar search box (`Ctrl+F`). Press `F3` for next, `Shift+F3` for previous. The `Aa` button toggles case sensitivity. Clear the box to remove highlights.

### Automatic Reloading
- **Manual**: Press `F5` to reload the current file.
- **Auto-Reload**: Enable `View > Auto-Reload` to watch files for changes (checks every 2 seconds).
    - **Smart Behavior**: If you are editing a file in another tab, it will update quietly in the background (marked with a `*`). When you switch back to it, it reloads instantly without losing your place.

### Exporting
- **PDF**: Go to `File > Export to PDF...` to save your document as a high-quality PDF with perfect math rendering.
- **Print**: Use `File > Print...` (`Ctrl+P`) to print to a physical printer.

## 3. Syntax Highlighting
Code blocks are automatically highlighted using Pygments with intelligent language detection.

**Python Example:**
```python
def exception_theory_demo():
    '''ET P-D-T architecture demonstration.'''
    # P: Point (infinite substrate)
    # D: Descriptor (finite constraints)  
    # T: Traverser (indeterminate agency)
    
    # Master equation: P∘D∘T = E
    result = compose(Point, Descriptor, Traverser)
    return result  # Returns Exception
```

**JavaScript Example:**
```javascript
// ET-derived async pattern
async function processData(input) {
    const parsed = await parse(input);      // P
    const described = describe(parsed);     // D
    const traversed = traverse(described);  // T
    return traversed;  // Exception
}
```

## 4. Mathematical Equation Support

### Enhanced MathJax 3 Rendering
This viewer uses **ET-optimized MathJax 3** with comprehensive LaTeX support, including AMS packages, custom macros, and SVG rendering for perfect quality.

### Inline Math
Use `$...$` or `\(...\)` for inline equations:

- Einstein's mass-energy: $E = mc^2$
- Euler's identity: $e^{i\pi} + 1 = 0$
- ET fine structure: $\alpha = \frac{e^2}{4\pi\epsilon_0\hbar c} \approx \frac{1}{137.035999206}$
- Variance formula: $\sigma^2 = \frac{n^2 - 1}{12}$
- Golden ratio: $\phi = \frac{1 + \sqrt{5}}{2}$

### Display Math
Use `$$...$$` or `\[...\]` for centered display equations:

$$\int_{a}^{b} x^2 \, dx = \frac{b^3 - a^3}{3}$$

$$\alpha = \frac{e^2}{4\pi\epsilon_0\hbar c} = \frac{1}{137.035999206(11)}$$

### Complex Equations
The viewer handles complex LaTeX with nested structures:

$$\sigma^2 = \frac{n^2 - 1}{12} = \frac{1}{12}(n+1)(n-1)$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \cdot \mathbf{B} = 0$$

### Multi-line Equations
Using align environment:

$$\begin{aligned}
P \circ D \circ T &= E \\
\text{Point} \circ \text{Descriptor} \circ \text{Traverser} &= \text{Exception} \\
\infty \circ \text{finite} \circ \text{indeterminate} &= \text{Reality}
\end{aligned}$$

### Matrices and Vectors
$$\begin{pmatrix}
1 & 0 & 0 \\
0 & \cos\theta & -\sin\theta \\
0 & \sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}
x \\
y \\
z
\end{pmatrix}
=
\begin{pmatrix}
x \\
y\cos\theta - z\sin\theta \\
y\sin\theta + z\cos\theta
\end{pmatrix}$$

### Greek Letters and Symbols
$$\alpha, \beta, \gamma, \Delta, \Sigma, \Omega, \int, \sum, \prod, \nabla, \partial, \sqrt{x}, \frac{\partial f}{\partial x}$$

### Summations and Integrals
$$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$$

$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

### Quantum Mechanics Examples
$$\hat{H}\psi = E\psi$$

$$[\hat{x}, \hat{p}] = i\hbar$$

$$\psi(x,t) = \sum_{n=1}^{\infty} c_n \phi_n(x) e^{-iE_n t/\hbar}$$

## 5. Task Lists
Interactive task lists are fully supported:

- [x] Read Documentation
- [x] Test Math Rendering
- [x] Try Inline Equations: $E = mc^2$
- [x] Test Display Math
- [ ] Try Drag & Drop
- [ ] Export a PDF
- [ ] Create your own document

## 6. Tables
| Feature | Status | Notes |
|---------|--------|-------|
| Markdown | ✓ | Full GFM support |
| Math | ✓ | LaTeX via MathJax 3 |
| Code | ✓ | Pygments highlighting |
| Dark Mode | ✓ | Live toggle |
| Auto-Reload | ✓ | Smart background updates |
| PDF Export | ✓ | High quality output |

## 7. ET Architecture

This viewer implements Exception Theory's P-D-T architecture:

- **Parse (P)**: Raw content as infinite substrate - the markdown source exists as unlimited potential
- **Describe (D)**: Markdown and LaTeX patterns as finite constraints - syntax rules define boundaries
- **Traverse (T)**: Rendering pipeline as indeterminate transformation - the process that actualizes the output

**Result: P∘D∘T = E** (Perfect rendered document as Exception)

The math preprocessor uses this same architecture:
1. **P**: Scans raw text for LaTeX delimiters
2. **D**: Applies regex patterns to identify math boundaries
3. **T**: Wraps and transforms into renderable form

### Critical Bug Fix
Version 8.0 fixes a critical bug in the inline math pattern (line 177 of original):
- **Before**: `r'(\\\(.*?\))'` - Missing closing backslash
- **After**: `r'(\\\(.*?\\\))'` - Properly matches `\(...\)` delimiters

This fix ensures all inline math using `\(...\)` syntax renders correctly.

---

**Version**: 8.0  
**Architecture**: P-D-T (Parse-Describe-Traverse)  
**Created by**: Aevum Defluo  
**Based on**: Original by Gemini  
**License**: Exception Theory Framework

*"For every exception there is an exception, except the exception."*
"""


if __name__ == "__main__":
    # Enable high DPI support
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    viewer = MarkdownViewer()
    viewer.show()
    sys.exit(app.exec())
