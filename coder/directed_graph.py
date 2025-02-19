"""
Created by Analitika at 18/02/2025
contact@analitika.fr

Improved version:
- Enhanced AST traversal to capture qualified names for functions and classes.
- Tracks function calls in their proper context.
- Extracts class inheritances.
- Adds type annotations to graph nodes and edges.
"""

# External imports
import os
import ast
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

# Internal imports
from coder.settings import DATA_FOLDER


class DependencyAnalyzer(ast.NodeVisitor):
    """
    Analyzes a single Python file to extract:
      - Imports
      - Function and class definitions (with qualified names)
      - Function calls (mapped to the calling context)
      - Class inheritance relationships
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.imports = set()
        self.functions = set()  # stores qualified function names
        self.classes = set()  # stores qualified class names
        self.calls = defaultdict(set)  # mapping: caller -> set(callees)
        self.inheritances = set()  # set of (child, parent) tuples

        # Stacks to keep track of current scope
        self.current_function_stack = []  # for tracking function/method context
        self.class_stack = []  # for tracking nested classes

    def visit_Import(self, node):
        """Capture import statements like 'import module'."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Capture imports from modules like 'from module import ...'."""
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """
        Capture a class definition.
        - Determine its qualified name (e.g., Outer.Inner).
        - Extract base classes for inheritance relationships.
        """
        # Compute the qualified class name based on the current class stack.
        qualified_class = (
            ".".join(self.class_stack + [node.name]) if self.class_stack else node.name
        )
        self.classes.add(qualified_class)

        # Process base classes to capture inheritance info.
        for base in node.bases:
            base_name = self._get_name_from_node(base)
            if base_name:
                self.inheritances.add((qualified_class, base_name))

        # Push this class to the class stack, visit its body, then pop.
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        """
        Capture a function or method definition.
        - Uses the class stack to qualify method names (e.g., ClassName.method).
        - Tracks the current function context to attribute calls correctly.
        """
        if self.class_stack:
            qualified_name = f"{'.'.join(self.class_stack)}.{node.name}"
        else:
            qualified_name = node.name

        self.functions.add(qualified_name)
        self.current_function_stack.append(qualified_name)
        self.generic_visit(node)
        self.current_function_stack.pop()

    def visit_Call(self, node):
        """
        Capture a function call.
        - Determines the callee name using _get_name_from_node.
        - Attributes the call to the current function context or '<module>' if outside any function.
        """
        callee = self._get_name_from_node(node.func)
        if callee:
            # If inside a function/method, attribute the call to it; otherwise, to module-level.
            caller = (
                self.current_function_stack[-1]
                if self.current_function_stack
                else "<module>"
            )
            self.calls[caller].add(callee)
        self.generic_visit(node)

    def _get_name_from_node(self, node):
        """
        Helper method to extract a name from different node types:
         - For a Name node, return its identifier.
         - For an Attribute node, recursively build the full attribute chain.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name_from_node(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return None

    def analyze(self):
        """
        Parses the file and returns a dictionary of extracted dependencies.
        Keys include: imports, functions, classes, calls, and inheritances.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=self.file_path)
        except SyntaxError as e:
            print(f"SyntaxError in {self.file_path}: {e}")
            return {}

        self.visit(tree)
        return {
            "imports": self.imports,
            "functions": self.functions,
            "classes": self.classes,
            "calls": dict(self.calls),
            "inheritances": self.inheritances,
        }


def analyze_project(directory):
    """
    Analyzes all Python files in a given directory to build a dependency graph.

    Returns:
        - dependency_graph: A directed graph (networkx.DiGraph) with nodes and edges labeled by type.
        - all_dependencies: A mapping from each file to its extracted dependency information.
    """
    dependency_graph = nx.DiGraph()
    all_dependencies = {}

    # Recursively find all .py files in the directory.
    py_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith(".py")
    ]

    for file in tqdm(py_files, desc="Analyzing files"):
        analyzer = DependencyAnalyzer(file)
        deps = analyzer.analyze()
        if not deps:
            continue  # Skip files that failed to parse
        all_dependencies[file] = deps

        # Create nodes for functions, classes, and a module-level node.
        for func in deps.get("functions", []):
            node_id = f"{file}::{func}"
            dependency_graph.add_node(node_id, type="function")
        for cls in deps.get("classes", []):
            node_id = f"{file}::{cls}"
            dependency_graph.add_node(node_id, type="class")
        # Module-level node to capture calls made outside any function.
        dependency_graph.add_node(f"{file}::<module>", type="module")

        # Add nodes for imports.
        for imp in deps.get("imports", []):
            dependency_graph.add_node(imp, type="import")

        # Add edges for function/method calls.
        for caller, callees in deps.get("calls", {}).items():
            # Determine caller node ID: if the caller was defined in the file, use its qualified name.
            caller_node = (
                f"{file}::{caller}"
                if caller in deps.get("functions", [])
                or caller in deps.get("classes", [])
                else f"{file}::<module>"
            )
            for callee in callees:
                # If the callee is defined in the same file, prefix with file; otherwise, use as-is.
                if callee in deps.get("functions", []) or callee in deps.get(
                    "classes", []
                ):
                    callee_node = f"{file}::{callee}"
                elif callee in deps.get("imports", []):
                    callee_node = callee
                else:
                    # For external or unknown functions, add a node if needed.
                    callee_node = callee
                    if not dependency_graph.has_node(callee_node):
                        dependency_graph.add_node(callee_node, type="unknown")
                dependency_graph.add_edge(caller_node, callee_node, type="call")

        # Add edges for inheritance relationships.
        for child, parent in deps.get("inheritances", []):
            child_node = f"{file}::{child}"
            if parent in deps.get("classes", []):
                parent_node = f"{file}::{parent}"
            elif parent in deps.get("imports", []):
                parent_node = parent
            else:
                parent_node = parent
                if not dependency_graph.has_node(parent_node):
                    dependency_graph.add_node(parent_node, type="unknown")
            dependency_graph.add_edge(child_node, parent_node, type="inheritance")

    return dependency_graph, all_dependencies


def example(project_path: str) -> tuple:
    graph, dependencies = analyze_project(project_path)

    # Save the dependency graph for later use.
    filename = os.path.join(DATA_FOLDER, "dependencies_graph.gml")
    nx.write_gml(graph, filename)

    # Print a summary of the graph.
    print("Graph nodes:")
    for node, data in graph.nodes(data=True):
        print(f"{node.replace(project_path, '.')}: {data}")

    print("\nGraph edges:")
    for u, v, data in graph.edges(data=True):
        print(
            f"{u.replace(project_path, '.')} -> {v.replace(project_path, '.')}: {data}"
        )

    return graph, dependencies


def filter_graph(graph, kind="remove", node="unknown") -> nx.Graph:
    """
    keep or remove nodes and edges from a graph
    """
    if kind == "remove":
        graph.remove_nodes_from(
            [n for n in graph.nodes if graph.nodes[n]["type"] == node]
        )
        # Remove edges connected to the removed nodes
        for edge in graph.edges:
            if (
                graph.nodes[edge[0]].get("type") == node
                or graph.nodes[edge[1]].get("type") == node
            ):
                graph.remove_edges_from(edge)

    elif kind == "keep":
        graph.remove_nodes_from(
            [n for n in graph.nodes if graph.nodes[n]["type"] != node]
        )
        for edge in graph.edges:
            if (
                graph.nodes[edge[0]].get("type") == node
                and graph.nodes[edge[1]].get("type") == node
            ):
                graph.remove_edges_from(edge)

    return graph


def visualize_directed_graph_interactive(graph, project_path="", plt_funcs=False):
    """
    Visualize a code dependency graph interactively using Plotly.

    Args:
        graph (networkx.Graph): The dependency graph.
        project_path (str): The root folder path to remove from node labels.
        plt_funcs (bool): Charts only function-to-function relations.
    """
    import plotly.graph_objects as go

    if plt_funcs:
        # Create a subgraph with only function-to-function relationships
        d_graph = filter_graph(graph, "keep", "functions")
    else:
        d_graph = graph

    # Build new labels by removing the root folder from node names
    new_labels = {}
    for node in d_graph.nodes:
        label = str(node)
        if project_path:
            label = label.replace(os.path.normpath(project_path), "").lstrip("/\\")
        new_labels[node] = label

    # Assign colors based on node type
    node_colors = []
    color_map = {
        "function": "lightblue",
        "class": "lightgreen",
        "import": "orange",
        "module": "violet",
        "unknown": "gray",
    }

    for node in d_graph.nodes:
        node_type = d_graph.nodes[node].get("type", "unknown")
        node_colors.append(color_map.get(node_type, "gray"))

    # Generate node positions using force-directed layout
    pos = nx.spring_layout(d_graph, seed=42)

    # Create edge traces
    edge_traces = []
    for edge in d_graph.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1, color="gray"),
            mode="lines",
            hoverinfo="none",
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x, node_y, node_text, node_hover = [], [], [], []
    for node in d_graph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(new_labels[node])
        node_hover.append(f"Type: {d_graph.nodes[node].get('type', 'unknown')}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=15, color=node_colors, opacity=0.8, line=dict(width=2, color="black")
        ),
        text=node_text,
        hovertext=node_hover,
        hoverinfo="text",
        textposition="top center",
    )

    # Create interactive figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Interactive Code Dependency Graph",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    fig.show()

    return


def visualize_directed_graph(graph, project_path="", plt_funcs=False) -> None:
    """
    Visualize a code dependency graph with improved aesthetics.

    Args:
        graph (networkx.Graph): The dependency graph.
        project_path (str): The root folder path to remove from node labels.
        plt_funcs (bool): Charts only function to function relations
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if plt_funcs:
        # Create a subgraph that contains only nodes of type 'function'
        d_graph = nx.DiGraph()
        for node in graph.nodes:
            if graph.nodes[node].get("type") == "function":
                d_graph.add_node(node, **graph.nodes[node])

        # Add edges only if both the source and target are function nodes.
        for edge in graph.edges:
            if (
                graph.nodes[edge[0]].get("type") == "function"
                and graph.nodes[edge[1]].get("type") == "function"
            ):
                d_graph.add_edge(*edge, **graph.get_edge_data(*edge))
    else:
        d_graph = graph

    # Build new labels by removing the root folder from node names.
    new_labels = {}
    for node in d_graph.nodes:
        label = str(node)
        if project_path:
            # Normalize paths and remove root_folder if present.
            label = label.replace(os.path.normpath(project_path), "")
            # Remove leading path separators if any.
            label = label.lstrip("/\\")
        new_labels[node] = label

    # Assign colors based on node type.
    node_colors = []
    for node in d_graph.nodes:
        node_type = d_graph.nodes[node].get("type", "unknown")
        if node_type == "function":
            node_colors.append("lightblue")
        elif node_type == "class":
            node_colors.append("lightgreen")
        elif node_type == "import":
            node_colors.append("orange")
        elif node_type == "module":
            node_colors.append("violet")
        else:
            node_colors.append("gray")  # Default color for unknown nodes

    # Create the layout.
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(d_graph, seed=42)

    # Draw nodes, edges, and labels separately.
    nx.draw_networkx_nodes(
        d_graph, pos, node_size=2000, node_color=node_colors, alpha=0.9
    )
    nx.draw_networkx_edges(d_graph, pos, edge_color="gray", arrows=True)
    nx.draw_networkx_labels(
        d_graph, pos, labels=new_labels, font_size=9, font_weight="bold"
    )

    # Create legend handles for node types.
    legend_handles = [
        mpatches.Patch(color="lightblue", label="Function"),
        mpatches.Patch(color="lightgreen", label="Class"),
        mpatches.Patch(color="orange", label="Import"),
        mpatches.Patch(color="violet", label="Module"),
        mpatches.Patch(color="gray", label="Unknown"),
    ]
    plt.legend(handles=legend_handles, loc="best")

    # Final plot settings.
    plt.title("Code Dependency Graph")
    plt.axis("off")
    # plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    # Update project_path to point to your Python project's root directory.
    project_path = r"C:\Users\ecepeda\OneDrive - analitika.fr\Documentos\PROYECTOS\ANALITIKA\PycharmProjects\neural_networks\coder"
    graph, dependencies = example(project_path)
    visualize_directed_graph_interactive(graph, project_path, False)
    visualize_directed_graph_interactive(graph, project_path, True)
    graph = filter_graph(graph, "remove", "unknown")
    visualize_directed_graph_interactive(graph, project_path, False)
    # visualize_directed_graph(graph, project_path, False)
