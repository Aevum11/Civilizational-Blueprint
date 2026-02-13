"""
ETPL IDE Stage 2 Enhancements
Advanced error detection, code intelligence, import tracing optimization
Derived from ET: Enhanced D (constraint validation), refined T (agency intelligence)
"""

import sys
import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QSplitter, QToolButton, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
    QFrame
)
from PyQt5.QtGui import (
    QColor, QPen, QBrush, QFont, QTextBlockUserData
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject


class ErrorDetector:
    """
    Advanced error detection for ETPL code.
    D: Constraint violation detection (Eq 211: Gap identification)
    T: Semantic analysis agency (Rule 7)
    """
    
    def __init__(self):
        # ET-derived error patterns
        self.error_patterns = self._init_error_patterns()
        self.semantic_rules = self._init_semantic_rules()
    
    def _init_error_patterns(self) -> List[Dict[str, Any]]:
        """Initialize syntax error detection patterns (Eq 225: Symbol derivation)"""
        return [
            {
                'pattern': r'P\s+(\w+)\s*(?!=)',
                'message': 'P primitive requires assignment (P id = value)',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'D\s+(\w+)\s*(?!=)',
                'message': 'D primitive requires assignment (D id = λ params . body)',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'T\s+(\w+)\s*(?!=)',
                'message': 'T primitive requires assignment (T id = → path)',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'λ\s*(?![a-zA-Z_])',
                'message': 'Lambda requires parameter list (λ param .)',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'∞\s*(?!\()',
                'message': 'Infinite loop requires condition: ∞ (expr) (D bound)',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'\[0/0\]\s*(?![\w\[])',
                'message': 'Exception ground requires expression: [0/0] expr',
                'severity': 'warning',
                'type': 'semantic'
            },
            {
                'pattern': r'→\s*E\s*(?![a-zA-Z_])',
                'message': 'Exception handler requires handler expression',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'ψ\s*(?![\w\[])',
                'message': 'Quantum wave requires parameters: ψ params . body',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'∘\s*$',
                'message': 'Binding operator ∘ requires operands on both sides',
                'severity': 'error',
                'type': 'syntax'
            },
            {
                'pattern': r'manifold\s*(?!\[)',
                'message': 'Manifold requires bracket notation: manifold [elements]',
                'severity': 'error',
                'type': 'syntax'
            },
        ]
    
    def _init_semantic_rules(self) -> List[Dict[str, Any]]:
        """Initialize semantic validation rules (Eq 223: Descriptor completion)"""
        return [
            {
                'check': self._check_balanced_brackets,
                'message': 'Unbalanced brackets detected',
                'severity': 'error'
            },
            {
                'check': self._check_balanced_parens,
                'message': 'Unbalanced parentheses detected',
                'severity': 'error'
            },
            {
                'check': self._check_undefined_references,
                'message': 'Undefined identifier: {ref}',
                'severity': 'warning'
            },
            {
                'check': self._check_binding_consistency,
                'message': 'PDT binding requires all three primitives',
                'severity': 'warning'
            },
        ]
    
    def detect_errors(self, code: str) -> List[Dict[str, Any]]:
        """
        Detect all errors in code.
        Returns list of error dictionaries with line, column, message, severity
        """
        errors = []
        
        # Syntax pattern matching
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern_info in self.error_patterns:
                matches = re.finditer(pattern_info['pattern'], line)
                for match in matches:
                    errors.append({
                        'line': line_num,
                        'column': match.start(),
                        'message': pattern_info['message'],
                        'severity': pattern_info['severity'],
                        'type': pattern_info['type'],
                        'length': match.end() - match.start()
                    })
        
        # Semantic analysis
        for rule in self.semantic_rules:
            semantic_errors = rule['check'](code)
            if semantic_errors:
                for error in semantic_errors:
                    errors.append({
                        'line': error.get('line', 0),
                        'column': error.get('column', 0),
                        'message': rule['message'].format(**error),
                        'severity': rule['severity'],
                        'type': 'semantic',
                        'length': error.get('length', 1)
                    })
        
        return errors
    
    def _check_balanced_brackets(self, code: str) -> List[Dict[str, Any]]:
        """Check bracket balance (Eq 202: D constraint validation)"""
        errors = []
        stack = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for col, char in enumerate(line):
                if char == '[':
                    stack.append({'line': line_num, 'column': col, 'char': '['})
                elif char == ']':
                    if not stack:
                        errors.append({
                            'line': line_num,
                            'column': col,
                            'length': 1
                        })
                    else:
                        stack.pop()
        
        # Unclosed brackets
        for item in stack:
            errors.append({
                'line': item['line'],
                'column': item['column'],
                'length': 1
            })
        
        return errors
    
    def _check_balanced_parens(self, code: str) -> List[Dict[str, Any]]:
        """Check parentheses balance"""
        errors = []
        stack = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for col, char in enumerate(line):
                if char == '(':
                    stack.append({'line': line_num, 'column': col, 'char': '('})
                elif char == ')':
                    if not stack:
                        errors.append({
                            'line': line_num,
                            'column': col,
                            'length': 1
                        })
                    else:
                        stack.pop()
        
        for item in stack:
            errors.append({
                'line': item['line'],
                'column': item['column'],
                'length': 1
            })
        
        return errors
    
    def _check_undefined_references(self, code: str) -> List[Dict[str, Any]]:
        """Check for undefined identifier references (Eq 204: P state binding)"""
        errors = []
        defined = set()
        
        # First pass: collect definitions
        lines = code.split('\n')
        for line in lines:
            # P definitions
            p_defs = re.findall(r'P\s+(\w+)\s*=', line)
            defined.update(p_defs)
            
            # D definitions
            d_defs = re.findall(r'D\s+(\w+)\s*=', line)
            defined.update(d_defs)
            
            # T definitions
            t_defs = re.findall(r'T\s+(\w+)\s*=', line)
            defined.update(t_defs)
        
        # Second pass: check references
        for line_num, line in enumerate(lines, 1):
            # Find identifier references (not in definitions)
            if not re.match(r'\s*[PDT]\s+\w+\s*=', line):
                refs = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line)
                for ref in refs:
                    # Skip ET keywords and symbols
                    if ref not in ['P', 'D', 'T', 'E', 'manifold', 'sovereign_import',
                                  'sin', 'cos', 'tan', 'log', 'lim'] and ref not in defined:
                        col = line.find(ref)
                        if col >= 0:
                            errors.append({
                                'line': line_num,
                                'column': col,
                                'length': len(ref),
                                'ref': ref
                            })
        
        return errors
    
    def _check_binding_consistency(self, code: str) -> List[Dict[str, Any]]:
        """Check PDT binding consistency (Eq 186: Complete binding)"""
        errors = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Find binding operators
            if '∘' in line:
                # Count primitives in binding expression
                parts = line.split('∘')
                if len(parts) > 1:
                    # Check if we have proper PDT pattern
                    has_p = any('P ' in part or 'Point' in part for part in parts)
                    has_d = any('D ' in part or 'Descriptor' in part for part in parts)
                    has_t = any('T ' in part or 'Traverser' in part for part in parts)
                    
                    if (has_p or has_d or has_t) and not (has_p and has_d and has_t):
                        errors.append({
                            'line': line_num,
                            'column': line.find('∘'),
                            'length': len(line.strip())
                        })
        
        return errors


class CodeIntelligence:
    """
    Code completion and intelligence features.
    D: Available symbols as constraints (Eq 225)
    T: Completion suggestions as agency navigation
    """
    
    def __init__(self):
        self.keywords = self._init_keywords()
        self.symbols = self._init_symbols()
        self.templates = self._init_templates()
    
    def _init_keywords(self) -> List[str]:
        """Initialize ETPL keywords"""
        return [
            'P', 'D', 'T', 'E',
            'manifold', 'sovereign_import',
            'sin', 'cos', 'tan', 'log', 'lim',
        ]
    
    def _init_symbols(self) -> Dict[str, str]:
        """Initialize ET symbols with descriptions"""
        return {
            '∘': 'Binding operator - binds P ∘ D ∘ T',
            '→': 'Arrow/Path operator - defines traversal path',
            'λ': 'Lambda - defines descriptor function',
            '∞': 'Infinity - infinite loop with bound',
            'ψ': 'Psi - quantum wavefunction',
            'Ω': 'Omega - infinite set',
            'ℵ': 'Aleph - cardinality',
            'ℏ': 'Planck constant - quantum operations',
            '[0/0]': 'Exception ground - indeterminate form',
            '∫': 'Integral - continuous operation',
            '∑': 'Summation - discrete operation',
            '∏': 'Product - multiplicative operation',
            '√': 'Square root',
            '∇': 'Nabla - gradient operator',
        }
    
    def _init_templates(self) -> Dict[str, str]:
        """Initialize code templates"""
        return {
            'P': 'P {id} = {value}',
            'D': 'D {id} = λ {params} . {body}',
            'T': 'T {id} = → {path}',
            'loop': '∞ ({condition}) (D {bound})',
            'exception': 'T {id} = → {expr} → E {handler}',
            'quantum': 'ψ {params} . {body}',
            'manifold': 'manifold [{elements}]',
            'binding': '{P} ∘ {D} ∘ {T}',
        }
    
    def get_completions(self, code: str, cursor_pos: int) -> List[Dict[str, str]]:
        """
        Get code completion suggestions at cursor position.
        Returns list of completion items with text and description
        """
        completions = []
        
        # Get text before cursor
        before_cursor = code[:cursor_pos]
        
        # Get current word being typed
        current_word = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*$', before_cursor)
        current_word = current_word[0] if current_word else ''
        
        # Keyword completions
        for keyword in self.keywords:
            if keyword.startswith(current_word):
                completions.append({
                    'text': keyword,
                    'description': f'ETPL keyword: {keyword}',
                    'type': 'keyword'
                })
        
        # Symbol completions (if typing backslash or special char)
        if before_cursor.endswith('\\') or current_word == '':
            for symbol, desc in self.symbols.items():
                completions.append({
                    'text': symbol,
                    'description': desc,
                    'type': 'symbol'
                })
        
        # Template completions
        for name, template in self.templates.items():
            if name.startswith(current_word):
                completions.append({
                    'text': template,
                    'description': f'Template: {name}',
                    'type': 'template'
                })
        
        return completions


class EnhancedImportTracer:
    """
    Enhanced import chain tracing with graph visualization.
    P: Import graph substrate (Eq 161)
    D: Dependency constraints (Eq 217: Recursive discovery)
    T: Graph navigation agency
    """
    
    def __init__(self):
        self.graph: Dict[str, List[str]] = {}
        self.visited: Set[str] = set()
    
    def trace_full_chain(self, root_file: str) -> Dict[str, Any]:
        """
        Trace complete import chain with dependency graph.
        Returns graph structure and metrics
        """
        self.graph = {}
        self.visited = set()
        
        # Build dependency graph
        self._build_graph(root_file)
        
        # Calculate metrics (Eq 216: Cardinality)
        total_files = len(self.graph)
        total_edges = sum(len(deps) for deps in self.graph.values())
        max_depth = self._calculate_max_depth(root_file)
        
        # Detect cycles
        cycles = self._detect_cycles()
        
        return {
            'graph': self.graph,
            'root': root_file,
            'total_files': total_files,
            'total_dependencies': total_edges,
            'max_depth': max_depth,
            'cycles': cycles,
            'files': list(self.graph.keys())
        }
    
    def _build_graph(self, file_path: str):
        """Build dependency graph recursively (Eq 217)"""
        if file_path in self.visited:
            return
        
        self.visited.add(file_path)
        
        if not os.path.exists(file_path):
            return
        
        # Initialize node
        self.graph[file_path] = []
        
        # Parse imports
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Python imports
            import_pattern = r'(?:from\s+(\S+)\s+import|import\s+(\S+))'
            matches = re.finditer(import_pattern, content)
            
            for match in matches:
                module = match.group(1) or match.group(2)
                module_path = self._resolve_import_path(module, file_path)
                
                if module_path:
                    self.graph[file_path].append(module_path)
                    self._build_graph(module_path)
        
        except Exception:
            pass
    
    def _resolve_import_path(self, module: str, current_file: str) -> Optional[str]:
        """Resolve import module name to file path"""
        # Try relative to current file
        current_dir = os.path.dirname(current_file)
        
        # Convert module.submodule to path
        module_path = module.replace('.', os.sep)
        
        # Try .py file
        possible_paths = [
            os.path.join(current_dir, f'{module_path}.py'),
            os.path.join(current_dir, module_path, '__init__.py'),
        ]
        
        # Try sys.path
        for path in sys.path:
            possible_paths.extend([
                os.path.join(path, f'{module_path}.py'),
                os.path.join(path, module_path, '__init__.py'),
            ])
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _calculate_max_depth(self, root: str, current_depth: int = 0) -> int:
        """Calculate maximum dependency depth"""
        if root not in self.graph or not self.graph[root]:
            return current_depth
        
        max_child_depth = current_depth
        for child in self.graph[root]:
            if child != root:  # Avoid self-loops
                child_depth = self._calculate_max_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies (Eq 236: Cycle prevention)"""
        cycles = []
        
        def dfs(node: str, path: List[str], visited_in_path: Set[str]):
            if node in visited_in_path:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if cycle not in cycles:
                    cycles.append(cycle)
                return
            
            if node not in self.graph:
                return
            
            visited_in_path.add(node)
            path.append(node)
            
            for neighbor in self.graph[node]:
                dfs(neighbor, path.copy(), visited_in_path.copy())
        
        for node in self.graph:
            dfs(node, [], set())
        
        return cycles


class ImportGraphView(QGraphicsView):
    """
    Visual graph representation of import dependencies.
    P: Graph nodes/edges substrate
    D: Layout constraints
    T: Interactive navigation
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Graph layout parameters
        self.node_radius = 30
        self.level_spacing = 120
        self.node_spacing = 80
        
        # Node color scheme (ET-derived)
        self.color_root = QColor(200, 50, 200)  # Magenta for root
        self.color_normal = QColor(100, 150, 255)  # Blue for normal
        self.color_cycle = QColor(255, 100, 100)  # Red for cycles
    
    def draw_graph(self, graph_data: Dict[str, Any]):
        """
        Draw import dependency graph.
        Layout using ET-derived tree/graph structure (Eq 227: Manifold domain)
        """
        self.scene.clear()
        
        graph = graph_data['graph']
        root = graph_data['root']
        
        # Calculate node positions (tree layout)
        positions = self._calculate_layout(graph, root)
        
        # Draw edges first (so they appear behind nodes)
        for source, targets in graph.items():
            if source not in positions:
                continue
            
            source_pos = positions[source]
            
            for target in targets:
                if target not in positions:
                    continue
                
                target_pos = positions[target]
                
                # Check if edge is part of cycle
                is_cycle = any(source in cycle and target in cycle 
                             for cycle in graph_data.get('cycles', []))
                
                pen = QPen(self.color_cycle if is_cycle else QColor(150, 150, 150), 2)
                if is_cycle:
                    pen.setStyle(Qt.DashLine)
                
                line = QGraphicsLineItem(
                    source_pos[0], source_pos[1],
                    target_pos[0], target_pos[1]
                )
                line.setPen(pen)
                self.scene.addItem(line)
        
        # Draw nodes
        for file_path, pos in positions.items():
            is_root = (file_path == root)
            in_cycle = any(file_path in cycle for cycle in graph_data.get('cycles', []))
            
            # Determine color
            if is_root:
                color = self.color_root
            elif in_cycle:
                color = self.color_cycle
            else:
                color = self.color_normal
            
            # Draw node circle
            ellipse = QGraphicsEllipseItem(
                pos[0] - self.node_radius,
                pos[1] - self.node_radius,
                self.node_radius * 2,
                self.node_radius * 2
            )
            ellipse.setBrush(QBrush(color))
            ellipse.setPen(QPen(Qt.white, 2))
            self.scene.addItem(ellipse)
            
            # Draw label (filename only)
            filename = os.path.basename(file_path)
            text = QGraphicsTextItem(filename)
            text.setPos(pos[0] - 40, pos[1] + self.node_radius + 5)
            text.setDefaultTextColor(Qt.white)
            font = QFont("Arial", 8)
            text.setFont(font)
            self.scene.addItem(text)
        
        # Fit in view
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def _calculate_layout(self, graph: Dict[str, List[str]], 
                         root: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using tree layout algorithm"""
        positions = {}
        
        # Level-order traversal for tree layout
        levels: Dict[int, List[str]] = {0: [root]}
        visited = {root}
        current_level = 0
        
        # Build levels
        while current_level in levels:
            next_level = []
            for node in levels[current_level]:
                if node in graph:
                    for child in graph[node]:
                        if child not in visited:
                            next_level.append(child)
                            visited.add(child)
            
            if next_level:
                levels[current_level + 1] = next_level
                current_level += 1
            else:
                break
        
        # Calculate positions
        max_width = max(len(nodes) for nodes in levels.values())
        total_width = max_width * self.node_spacing
        
        for level, nodes in levels.items():
            y = level * self.level_spacing
            level_width = len(nodes) * self.node_spacing
            start_x = (total_width - level_width) / 2
            
            for i, node in enumerate(nodes):
                x = start_x + i * self.node_spacing
                positions[node] = (x, y)
        
        return positions


class ErrorListWidget(QListWidget):
    """
    Widget for displaying detected errors and warnings.
    P: Error list substrate
    T: Error navigation agency
    """
    
    error_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Connect selection signal
        self.itemClicked.connect(self._on_item_clicked)
    
    def set_errors(self, errors: List[Dict[str, Any]]):
        """Display list of errors"""
        self.clear()
        
        for error in errors:
            severity = error.get('severity', 'error')
            line = error.get('line', 0)
            message = error.get('message', 'Unknown error')
            
            # Format item text
            text = f"[{severity.upper()}] Line {line}: {message}"
            
            item = QListWidgetItem(text)
            
            # Color code by severity
            if severity == 'error':
                item.setForeground(QColor(255, 100, 100))
            elif severity == 'warning':
                item.setForeground(QColor(255, 200, 0))
            else:
                item.setForeground(QColor(150, 150, 150))
            
            # Store error data
            item.setData(Qt.UserRole, error)
            
            self.addItem(item)
    
    def _on_item_clicked(self, item: QListWidgetItem):
        """Emit signal when error is clicked"""
        error_data = item.data(Qt.UserRole)
        if error_data:
            self.error_selected.emit(error_data)


def create_stage2_enhancements():
    """
    Factory function to create Stage 2 enhancement components.
    Returns dict of enhancement objects for integration into main IDE.
    """
    return {
        'error_detector': ErrorDetector(),
        'code_intelligence': CodeIntelligence(),
        'import_tracer': EnhancedImportTracer(),
        'import_graph_view': ImportGraphView,
        'error_list_widget': ErrorListWidget,
    }


# Integration helpers for main IDE
def integrate_stage2_into_ide(main_window):
    """
    Integrate Stage 2 enhancements into existing IDE window.
    This function modifies the main window to add Stage 2 features.
    """
    
    # Create enhancement components
    enhancements = create_stage2_enhancements()
    
    # Add error detector
    main_window.error_detector = enhancements['error_detector']
    
    # Add code intelligence
    main_window.code_intelligence = enhancements['code_intelligence']
    
    # Add enhanced import tracer
    main_window.enhanced_import_tracer = enhancements['import_tracer']
    
    # Add error list widget to UI
    error_dock = QDockWidget("Errors & Warnings", main_window)
    error_list = ErrorListWidget()
    error_dock.setWidget(error_list)
    main_window.addDockWidget(Qt.BottomDockWidgetArea, error_dock)
    main_window.error_list = error_list
    
    # Add import graph view
    graph_dock = QDockWidget("Import Dependency Graph", main_window)
    graph_view = ImportGraphView()
    graph_dock.setWidget(graph_view)
    main_window.addDockWidget(Qt.RightDockWidgetArea, graph_dock)
    main_window.import_graph_view = graph_view
    
    # Connect error selection to editor
    def on_error_selected(error_data):
        editor = main_window._get_current_editor()
        cursor = editor.textCursor()
        
        # Move to error line
        line = error_data.get('line', 1)
        cursor.movePosition(cursor.Start)
        cursor.movePosition(cursor.Down, cursor.MoveAnchor, line - 1)
        
        # Select error region if length provided
        length = error_data.get('length', 0)
        column = error_data.get('column', 0)
        if length > 0:
            cursor.movePosition(cursor.Right, cursor.MoveAnchor, column)
            cursor.movePosition(cursor.Right, cursor.KeepAnchor, length)
        
        editor.setTextCursor(cursor)
        editor.setFocus()
    
    error_list.error_selected.connect(on_error_selected)
    
    # Add real-time error detection
    def on_text_changed():
        editor = main_window._get_current_editor()
        code = editor.get_text()
        
        # Detect errors
        errors = main_window.error_detector.detect_errors(code)
        main_window.error_list.set_errors(errors)
    
    # Connect to all editors (needs to be done for each new tab)
    main_window._original_new_editor_tab = main_window._new_editor_tab
    
    def enhanced_new_editor_tab(title="Untitled", content=""):
        editor = main_window._original_new_editor_tab(title, content)
        editor.textChanged.connect(on_text_changed)
        return editor
    
    main_window._new_editor_tab = enhanced_new_editor_tab
    
    # Enhanced import tracing
    main_window._original_trace_imports = main_window._trace_imports
    
    def enhanced_trace_imports():
        editor = main_window._get_current_editor()
        
        if not editor.file_path:
            main_window.console.append_error("Please save file first")
            return
        
        main_window.console.append_info("=" * 60)
        main_window.console.append_info("TRACING IMPORT CHAIN (ENHANCED)...")
        
        try:
            # Use enhanced tracer
            graph_data = main_window.enhanced_import_tracer.trace_full_chain(editor.file_path)
            
            # Display in tree
            main_window.import_tree.set_import_chain(graph_data['files'])
            
            # Display in graph view
            main_window.import_graph_view.draw_graph(graph_data)
            
            # Report metrics
            main_window.console.append_success(f"Import analysis complete!")
            main_window.console.append_info(f"Total files: {graph_data['total_files']}")
            main_window.console.append_info(f"Total dependencies: {graph_data['total_dependencies']}")
            main_window.console.append_info(f"Maximum depth: {graph_data['max_depth']}")
            
            if graph_data['cycles']:
                main_window.console.append_output(f"⚠ WARNING: {len(graph_data['cycles'])} circular dependencies detected", "orange")
                for cycle in graph_data['cycles']:
                    cycle_str = ' → '.join(os.path.basename(f) for f in cycle)
                    main_window.console.append_output(f"  Cycle: {cycle_str}", "orange")
            
            main_window.status_bar.showMessage(
                f"Traced {graph_data['total_files']} files, "
                f"{graph_data['total_dependencies']} dependencies"
            )
            
        except Exception as e:
            main_window.console.append_error(f"Enhanced import tracing failed: {str(e)}")
            main_window.status_bar.showMessage("Import tracing failed")
    
    main_window._trace_imports = enhanced_trace_imports
    
    return main_window
