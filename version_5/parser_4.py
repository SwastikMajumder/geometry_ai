import copy
from lark import Lark, Tree


# Basic data structure, which can nest to represent math equations
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

# convert string representation into tree
def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") # add a dummy node
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') # count the spaces, which is crucial information in a string representation
        node_name = line.strip() # remove spaces, when putting it in the tree form
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] # remove dummy node

# convert tree into string representation
def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) # spacings
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) # one node in one line
        return result
    return recursive_str(node)

grammar = r"""
?start: equation

?equation: expr "=" expr -> eq | expr // Handle optional equality operator.

?expr: base
     | expr "+" expr   -> add
     | expr "-" expr   -> sub

?base: INTEGER                -> number
     | "angle(" LETTERS ")"  -> angle  // Handle angle function with arguments.
     | "line(" LETTERS ")"   -> line  // Handle line function with arguments.
     | "parallel(" args ")"  -> parallel  // Handle parallel function with arguments.
     | "exist(" args ")"     -> exist  // Handle exist function with arguments.
     | "(" expr ")"          -> paren  // Parenthesized expressions.
     | "congruent(" args ")" -> congruent
     | "triangle(" LETTERS ")" -> triangle

?args: expr ("," expr)*  -> func_args  // Handle comma-separated function arguments.

LETTERS: /[A-Z]+/            // Sequence of letters without spaces.
INTEGER: /[+-]?\d+/

%import common.WS_INLINE
%ignore WS_INLINE
"""




def add_prefix_to_angle_children(node):
    """
    Recursively add the prefix 'd_' to all children of nodes with the name 'angle'.
    
    :param node: The root node of the tree or subtree to process.
    :return: The updated tree with prefixes added.
    """
    # If the node's name is 'angle', add the 'd_' prefix to its children
    if node.name in {"f_triangle", "f_angle"}:
        # Create new TreeNode instances for children with 'd_' prefix
        child_name = node.children[0].name
        a, b, c = "d_"+child_name[0], "d_"+child_name[1], "d_"+child_name[2]
        a, b, c = tree_form(a), tree_form(b), tree_form(c)
        node.children = [a,b,c]
    elif node.name == "f_line":
        child_name = node.children[0].name
        a, b = "d_"+child_name[0], "d_"+child_name[1]
        a, b = tree_form(a), tree_form(b)
        node.children = [a,b]
        
    coll = TreeNode(node.name, [])
    for child in node.children:
        coll.children.append(add_prefix_to_angle_children(child))
    return node

# Create a parser
parser = Lark(grammar, start='start', parser='lalr')

# Example equation to parse
def take_input(equation):#"sin(3)+1+2"
  #equation = equation.replace(" ", "").replace(",", "")
  # Parse the equation
  parse_tree = parser.parse(equation)

  def convert_to_treenode(parse_tree):
      def tree_to_treenode(tree):
          if isinstance(tree, Tree):
              node = TreeNode(tree.data)
              node.children = [tree_to_treenode(child) for child in tree.children]
              return node
          else:  # Leaf node
              return TreeNode(str(tree))

      return tree_to_treenode(parse_tree)

  def replace(equation, find, r):
    if str_form(equation) == str_form(find):
      return r
    col = TreeNode(equation.name, [])
    for child in equation.children:
      col.children.append(replace(child, find, r))
    return col

  def rename_internal_undesired_nodes(node, parent_name=None):
    # Define the set of relevant node names
    relevant_nodes = {"triangle", "congruent", "angle", "add", "sub", "line", "parallel", "exist", "and", "mul", "eq"}

    # Rename only internal undesired nodes
    if node.name not in relevant_nodes:
        if parent_name and node.children:  # Rename only if the node is internal (has children)
            node.name = parent_name
        # Process each child recursively
        cleaned_children = [rename_internal_undesired_nodes(child, node.name) for child in node.children]
        return TreeNode(node.name, cleaned_children)
    else:
        # Process each child recursively
        cleaned_children = [rename_internal_undesired_nodes(child, node.name) for child in node.children]
        return TreeNode(node.name, cleaned_children)

  def remove_duplicates(node):
    # Recursively process and clean the children nodes
    cleaned_children = [remove_duplicates(child) for child in node.children]

    # If the parent node has only one child and the parent's name matches the child's name
    if len(cleaned_children) == 1 and cleaned_children[0].name == node.name:
        return cleaned_children[0]  # Replace the parent node with its child

    # Otherwise, return the current node with its cleaned children
    return TreeNode(node.name, cleaned_children)

  def remove_past(node):
      if len(node.children) == 1 and node.name not in {"triangle", "congruent", "angle", "add", "mul", "sub", "line", "parallel", "exist", "and", "eq"}:
          return remove_past(node.children[0])
      coll = TreeNode(node.name, [])
      for child in node.children:
          coll.children.append(remove_past(child))
      return coll
  # Convert and print TreeNode structure
  tree_node = convert_to_treenode(parse_tree)
  #print(str_form(tree_node))
  tree_node = remove_past(tree_node)
  tree_node = rename_internal_undesired_nodes(tree_node)
  #print(str_form(tree_node))
  tree_node = remove_duplicates(tree_node)
  #print(str_form(tree_node))
  tree_node = str_form(tree_node)
  #print(tree_node)
  def name_replace(tree_node):
      coll = TreeNode(tree_node.name, [])
      if tree_node.name in {"triangle", "mul", "congruent", "angle", "add", "sub", "line", "parallel", "exist", "and", "eq"}:
          coll = TreeNode("f_" + tree_node.name, [])
      for child in tree_node.children:
          coll.children.append(name_replace(child))
      return coll
  tree_node = str_form(name_replace(tree_form(tree_node)))
  tree_node = tree_form(tree_node)
  tree_node = add_prefix_to_angle_children(tree_node)
  for i in range(360,-1,-1):
      tree_node = replace(tree_node, tree_form(str(i)), tree_form("d_"+str(i)))
  tree_node = str_form(tree_node).replace("true", "d_true")
  
  return tree_node
