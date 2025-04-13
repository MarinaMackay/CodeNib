import re
import json
import networkx as nx
from collections import defaultdict

class SCIPGraphDecoder:
    def __init__(self, index_file_path):
        self.index_file_path = index_file_path
        self.graph = nx.DiGraph()
        self.current_file = None
        self.current_scope = None
        self.scope_stack = []
        # Store line ranges for symbols
        self.symbol_ranges = {}

    def decode(self):
        with open(self.index_file_path, 'r') as f:
            content = f.read()
        
        # Parse documents
        document_blocks = re.findall(r'documents\s*{(.*?)(?=documents\s*{|$)', content, re.DOTALL)
        
        for document in document_blocks:
            self._process_document(document)
        
        return self.graph
    
    def _process_document(self, document_text):
        # Extract file path
        file_match = re.search(r'relative_path:\s*"([^"]+)"', document_text)
        if not file_match:
            return
        
        file_path = file_match.group(1)
        self.current_file = file_path
        
        # Add file node
        self.graph.add_node(file_path, type='file')
        self.current_scope = file_path
        self.scope_stack = [file_path]
        
        # Process occurrences
        occurrences = re.findall(r'occurrences\s*{(.*?)}', document_text, re.DOTALL)
        for occurrence in occurrences:
            self._process_occurrence(occurrence)
    
    def _process_occurrence(self, occurrence_text):
        # Skip local symbols
        if 'local' in occurrence_text:
            return
        
        # Skip stdlib symbols
        if 'python-stdlib' in occurrence_text:
            return
        
        # Extract ranges
        ranges = re.findall(r'range:\s*(\d+)', occurrence_text)
        if len(ranges) < 3:
            return
        
        line = int(ranges[0])
        start_col = int(ranges[1])
        end_col = int(ranges[2])
        
        # Extract symbol
        symbol_match = re.search(r'symbol:\s*"([^"]+)"', occurrence_text)
        if not symbol_match:
            return
        
        symbol = symbol_match.group(1)
        
        # Extract symbol_roles
        symbol_roles_match = re.search(r'symbol_roles:\s*(\d+)', occurrence_text)
        if not symbol_roles_match:
            return
        
        symbol_roles = int(symbol_roles_match.group(1))
        
        # Extract enclosing range if available
        enclosing_ranges = re.findall(r'enclosing_range:\s*(\d+)', occurrence_text)
        
        # Process the symbol
        self._process_symbol(symbol, line, start_col, end_col, symbol_roles, enclosing_ranges)
    
    def _process_symbol(self, symbol, line, start_col, end_col, symbol_roles, enclosing_ranges):
        # Skip function arguments (symbols ending with .(xxx))
        if re.search(r'\.\([^)]+\)$', symbol):
            return
        
        # Parse the symbol
        match = re.search(r'`?([^`]+)`?/([^.]+)(?:\.|\(|#)', symbol)
        if not match:
            return
        
        module_path = match.group(1)
        symbol_name = match.group(2)
        
        # Clean up the symbol for node name
        cleaned_symbol = re.sub(r'scip-python python \. [a-f0-9]+ `?', '', symbol)
        cleaned_symbol = re.sub(r'`', '', cleaned_symbol)
        
        # Handle __init__ symbols - convert to file reference
        if '/__init__' in cleaned_symbol:
            # Extract the module path and use it as the target
            module_match = re.search(r'(.+)/(?:__init__)', cleaned_symbol)
            if module_match:
                module_path = module_match.group(1)
                file_path = module_path.replace('.', '/') + '.py'
                
                # If this is a reference, point to the file instead
                if symbol_roles == 8:
                    self.graph.add_edge(self.current_scope, file_path, type='reference')
                return
        
        # Update current scope if this is a definition with enclosing range
        if symbol_roles == 1 and enclosing_ranges and len(enclosing_ranges) >= 4:
            scope_start_line = int(enclosing_ranges[0])
            scope_end_line = int(enclosing_ranges[2])
            
            # Store symbol range
            self.symbol_ranges[cleaned_symbol] = (scope_start_line, scope_end_line)
            
            # Update current scope
            self.current_scope = cleaned_symbol
            self.scope_stack.append(cleaned_symbol)
            
            # Add symbol node
            self.graph.add_node(cleaned_symbol, type='symbol', file=self.current_file,
                               start_line=scope_start_line, end_line=scope_end_line)
            
            # Add 'contain' edge from file to symbol (without attributes)
            parent_scope = self.scope_stack[-2] if len(self.scope_stack) > 1 else self.current_file
            self.graph.add_edge(parent_scope, cleaned_symbol, type='contain')
            
        # Handle reference (symbol_roles == 8)
        elif symbol_roles == 8:
            # If the symbol doesn't exist, create it without range info (to be filled later)
            if cleaned_symbol not in self.graph:
                self.graph.add_node(cleaned_symbol, type='symbol', file=module_path)
            
            # Add 'reference' edge from current scope to referenced symbol (without attributes)
            self.graph.add_edge(self.current_scope, cleaned_symbol, type='reference')
    
    def save_graph(self, output_path):
        # Convert graph to JSON
        data = {
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in self.graph.nodes(data=True):
            node_data = {'id': node, **attrs}
            data['nodes'].append(node_data)
        
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {'source': source, 'target': target, 'type': attrs.get('type')}
            data['edges'].append(edge_data)
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

# Usage
if __name__ == "__main__":
    decoder = SCIPGraphDecoder("index.decoded")
    graph = decoder.decode()
    decoder.save_graph("scip_graph.json")
    
    # Print some stats
    print(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Count node types
    file_nodes = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get('type') == 'file')
    symbol_nodes = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get('type') == 'symbol')
    
    # Count edge types
    contain_edges = sum(1 for _, _, attrs in graph.edges(data=True) if attrs.get('type') == 'contain')
    reference_edges = sum(1 for _, _, attrs in graph.edges(data=True) if attrs.get('type') == 'reference')
    
    print(f"Files: {file_nodes}, Symbols: {symbol_nodes}")
    print(f"Contain edges: {contain_edges}, Reference edges: {reference_edges}")