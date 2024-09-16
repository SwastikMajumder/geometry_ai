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

?equation: expr ("=" expr)?   -> eq

?expr: sign_expr
     | expr "+" sign_expr   -> add
     | expr "-" sign_expr   -> sub

?sign_expr: "-" term   -> neg
          | term

?term: factor
     | term "*" factor  -> mul
     | term "/" factor  -> div

?factor: base
       | factor "^" base  -> pow

?base: NUMBER              -> number
     | FUNC_NAME "(" expr ")" -> func
     | VARIABLE            -> variable
     | "(" expr ")"        -> paren

FUNC_NAME: "sin" | "cos" | "tan" | "log" | "sqrt" | "int" | "dif"
VARIABLE: "x" | "y" | "z" | "w"

NUMBER: INTEGER | FLOAT

INTEGER: /[+-]?\d+/
FLOAT: /[+-]?\d+\.\d+/

%import common.WS_INLINE
%ignore WS_INLINE
"""



# Create a parser
parser = Lark(grammar, start='start', parser='lalr')

# Example equation to parse
def take_input(equation):#"sin(3)+1+2"

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

  # fancy print
  def string_equation_helper(equation_tree):
      if equation_tree.children == []:
          return equation_tree.name # leaf node
      s = "(" # bracket
      if len(equation_tree.children) == 1:
          s = equation_tree.name[2:] + s
      sign = {"f_add": "+", "f_mul": "*", "f_pow": "^", "f_div": "/", "f_int": ",", "f_sub": "-", "f_dif": "?", "f_sin": "?", "f_cos": "?", "f_tan": "?", "f_eq": "=", "f_sqt": "?"} # operation symbols
      for child in equation_tree.children:
          s+= string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
      s = s[:-1] + ")"
      return s

  # fancy print main function
  def string_equation(eq): 
      eq = eq.replace("v_0", "x")
      eq = eq.replace("v_1", "y")
      eq = eq.replace("v_2", "z")
      eq = eq.replace("d_", "")
      
      return string_equation_helper(tree_form(eq))

  def replace(equation, find, r):
    if str_form(equation) == str_form(find):
      return r
    col = TreeNode(equation.name, [])
    for child in equation.children:
      col.children.append(replace(child, find, r))
    return col
  def remove_past(equation):
      if equation.name in {"number", "paren", "func", "variable"}:
          if len(equation.children) == 1:
            for index, child in enumerate(equation.children):
              equation.children[index] = remove_past(child)
            return equation.children[0]
          else:
            for index, child in enumerate(equation.children):
              equation.children[index] = remove_past(child)
            return TreeNode(equation.children[0].name, equation.children[1:])
      coll = TreeNode(equation.name, [])
      for child in equation.children:
          coll.children.append(remove_past(child))
      return coll
  # Convert and print TreeNode structure
  tree_node = convert_to_treenode(parse_tree)
  tree_node = remove_past(tree_node)
  tree_node = str_form(tree_node)
  #print(tree_node)
  for item in ["sub", "add", "sin", "cos", "tan", "mul", "int", "dif", "pow", "div", "eq"]:
    tree_node = tree_node.replace(str(item), "f_" + str(item))
  tree_node = tree_form(tree_node)
  for i in range(100,-1,-1):
    tree_node = replace(tree_node, tree_form(str(i)), tree_form("d_"+str(i)))
  for i in range(4):
    tree_node = replace(tree_node, tree_form(["x", "y", "z", "w"][i]), tree_form("v_"+str(i)))
  tree_node = str_form(tree_node)
  return tree_node
