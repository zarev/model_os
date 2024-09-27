# Libs
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Tree
from rich.emoji import Emoji
from rich.text import Text
# Locals
from utils.time import convert_millis, to_ISO

class BuildNodesTree(Widget):

    def __init__(self, build: dict) -> None:
        self.build = build
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Tree('KdMonoRepo/parametrised', id='tree', classes='with_title')

    def on_mount(self) -> None:
        exec_tree = self.query_one('#tree', Tree)
        exec_tree.border_title = ":evergreen_tree: Execution nodes tree"
        self.build['nodes'][0]['node'] = exec_tree.root.add(self.build['title'], expand=True)
        for idx in self.build['nodes']:
            exec_node=self.build['nodes'][idx]
            # Generate the node content
            result_symbol = Text(Emoji.replace(f":{exec_node['colour']}_circle:"))
            # Add exec_data to the tree graph
            tree_node_text = Text()
            tree_node_text.append(result_symbol)
            tree_node_text.append(' ')
            if exec_node['total_duration']:
                tree_node_text.append(f"{convert_millis(exec_node['total_duration'])}", "red")
                tree_node_text.append('|')
            tree_node_text.append(f"{convert_millis(exec_node['step_duration'])}", "blue")
            tree_node_text.append(' - ')
            tree_node_text.append(exec_node['name'])
            # Use the parent as the tree anchor
            tree_node = self.build['nodes'][exec_node['parent']]['node'].add(tree_node_text, expand=True)
            self.build['nodes'][idx]['node'] = tree_node
        exec_tree.show_root=False