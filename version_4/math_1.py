import copy
import numpy as np
import itertools
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

# Generate transformations of a given equation provided only one formula to do so
# We can call this function multiple times with different formulas, in case we want to use more than one
# This function is also responsible for computing arithmetic, pass do_only_arithmetic as True (others param it would ignore), to do so
def apply_individual_formula_on_given_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic=False, structure_satisfy=False):
    variable_list = {}
    
    def node_type(s):
        if s[:2] == "f_":
            return s
        else:
            return s[:2]
    def does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs):
        nonlocal variable_list
        # u can accept anything and p is expecting only integers
        # if there is variable in the formula
        if node_type(formula_lhs.name) in {"u_", "p_"}: 
            if formula_lhs.name in variable_list.keys(): # check if that variable has previously appeared or not
                return str_form(variable_list[formula_lhs.name]) == str_form(equation) # if yes, then the contents should be same
            else: # otherwise, extract the data from the given equation
                if node_type(formula_lhs.name) == "p_" and "v_" in str_form(equation): # if formula has a p type variable, it only accepts integers
                    return False
                variable_list[formula_lhs.name] = copy.deepcopy(equation)
                return True
        if equation.name != formula_lhs.name or len(equation.children) != len(formula_lhs.children): # the formula structure should match with given equation
            return False
        for i in range(len(equation.children)): # go through every children and explore the whole formula / equation
            if does_given_equation_satisfy_forumla_lhs_structure(equation.children[i], formula_lhs.children[i]) is False:
                return False
        return True
    if structure_satisfy:
      return does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs)
    # transform the equation as a whole aka perform the transformation operation on the entire thing and not only on a certain part of the equation
    def formula_apply_root(formula):
        nonlocal variable_list
        if formula.name in variable_list.keys():
            return variable_list[formula.name] # fill the extracted data on the formula rhs structure
        data_to_return = TreeNode(formula.name, None) # produce nodes for the new transformed equation
        for child in formula.children:
            data_to_return.children.append(formula_apply_root(copy.deepcopy(child))) # slowly build the transformed equation
        return data_to_return
    count_target_node = 1
    # try applying formula on various parts of the equation
    def formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic):
        nonlocal variable_list
        nonlocal count_target_node
        data_to_return = TreeNode(equation.name, children=[])
        variable_list = {}
        if do_only_arithmetic == False:
            if does_given_equation_satisfy_forumla_lhs_structure(equation, copy.deepcopy(formula_lhs)) is True: # if formula lhs structure is satisfied by the equation given
                count_target_node -= 1
                if count_target_node == 0: # and its the location we want to do the transformation on
                    return formula_apply_root(copy.deepcopy(formula_rhs)) # transform
        else: # perform arithmetic
            if len(equation.children) == 2 and all(node_type(item.name) == "d_" for item in equation.children): # if only numbers
                x = []
                for item in equation.children:
                    x.append(float(item.name[2:])) # convert string into a number
                if equation.name == "f_add":
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(sum(x))) # add all
                elif equation.name == "f_mul":
                    count_target_node -= 1
                    if count_target_node == 0:
                        p = 1
                        for item in x:
                            p *= item # multiply all
                        return TreeNode("d_" + str(p))
                elif equation.name == "f_pow" and x[1]>=2: # power should be two or a natural number more than two
                    count_target_node -= 1
                    if count_target_node == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
                elif equation.name == "f_sub":
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(x[0]-x[1]))
                elif equation.name == "f_div" and int(x[0]/x[1]) == x[0]/x[1]:
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(int(x[0]/x[1])))
        if node_type(equation.name) in {"d_", "v_"}: # reached a leaf node
            return equation
        for child in equation.children: # slowly build the transformed equation
            data_to_return.children.append(formula_apply_various_sub_equation(copy.deepcopy(child), formula_lhs, formula_rhs, do_only_arithmetic))
        return data_to_return
    cn = 0
    # count how many locations are present in the given equation
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    transformed_equation_list = []
    count_nodes(equation)
    for i in range(1, cn + 1): # iterate over all location in the equation tree
        count_target_node = i
        orig_len = len(transformed_equation_list)
        tmp = formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic)
        if str_form(tmp) != str_form(equation): # don't produce duplication, or don't if nothing changed because of transformation impossbility in that location
            transformed_equation_list.append(str_form(tmp)) # add this transformation to our list
    return transformed_equation_list 

# Function to generate neighbor equations
def generate_transformation(equation, file_name):
    input_f, output_f = return_formula_file(file_name) # load formula file
    transformed_equation_list = []
    for i in range(len(input_f)): # go through all formulas and collect if they can possibly transform
        transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(copy.deepcopy(equation)), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to generate neighbor equations
def generate_arithmetical_transformation(equation):
    transformed_equation_list = []
    transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), None, None, True, False) # perform arithmetic
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to read formula file
def return_formula_file(file_name):
    with open(file_name, 'r') as file:
      content = file.read()
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] # alternative formula lhs and then formula rhs
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f] # convert into tree form
    output_f = [tree_form(item) for item in output_f]
    return [input_f, output_f] # return

def search(equation, depth, file_list, auto_arithmetic=True, visited=None):
    final = []
    def search_helper(equation, depth, file_list, auto_arithmetic, visited):
        if depth == 0: # limit the search
            return None
        if visited is None:
            visited = set()
        final.append(equation)
        #print(string_equation(equation))
        if equation in visited:
            return None
        visited.add(equation)
        output =[]
        if file_list[0]:
          output += generate_transformation(equation, file_list[0])
        if auto_arithmetic:
          output += generate_arithmetical_transformation(equation)
        if len(output) > 0:
          output = [output[0]]
        else:
          if file_list[1]:
            output += generate_transformation(equation, file_list[1])
          if not auto_arithmetic:
            output += generate_arithmetical_transformation(equation)
          if file_list[2] and len(output) == 0:
              output += generate_transformation(equation, file_list[2])
        for i in range(len(output)):
            search_helper(output[i], depth-1, file_list, auto_arithmetic, visited) # recursively find even more equals
    search_helper(equation, depth, file_list, auto_arithmetic, visited)
    return final

def string_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name # leaf node
    s = "(" # bracket
    if len(equation_tree.children) == 1 or equation_tree.name in ["f_xangle", "f_xline", "f_angle", "f_line", "f_parallel"]:
        s = equation_tree.name[2:] + s
    sign = {"f_mul": "*", "f_add": "+", "f_and": "^", "f_eq": "=", "f_angle": "?", "f_xangle": "?", "f_parallel": ",", "f_xline": "?", "f_exist": "?", "f_line": "?"} # operation symbols
    for child in equation_tree.children:
        s+= string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
    s = s[:-1] + ")"
    return s

# fancy print main function
def string_equation(eq):
    eq = eq.replace("d_", "")
    return string_equation_helper(tree_form(eq)).replace("?", "")

def replace(equation, find, r):
  if str_form(equation) == str_form(find):
    return r
  col = TreeNode(equation.name, [])
  for child in equation.children:
    col.children.append(replace(child, find, r))
  return col

def flatten_tree(node):
    if not node.children:
        return node
    if node.name in ("f_add", "f_mul"):
        merged_children = []
        for child in node.children:
            flattened_child = flatten_tree(child)
            if flattened_child.name == node.name:
                merged_children.extend(flattened_child.children)
            else:
                merged_children.append(flattened_child)
        return TreeNode(node.name, merged_children)
    else:
        node.children = [flatten_tree(child) for child in node.children]
        return node

def flatten_tree_one_level(node):
    if not node.children:
        return node
    
    if node.name in ("f_add", "f_mul"):
        # Flatten only the immediate children
        merged_children = []
        for child in node.children:
            if child.name == node.name:
                # Merge the children of the child node into the current level
                merged_children.extend(child.children)
            else:
                # Otherwise, just add the child as is
                merged_children.append(child)
        
        # Return the node with its children flattened one level
        return TreeNode(node.name, merged_children)
    else:
        # For non-commutative operations, process the children recursively
        node.children = [flatten_tree_one_level(child) for child in node.children]
        return node
    
def break_equation(equation):
    sub_equation_list = [equation]
    equation = equation
    for child in equation.children: # breaking equation by accessing children
        sub_equation_list += break_equation(child) # collect broken equations
    return sub_equation_list

def identify_term(equation):
    pass
    
def unflatten(equation):
    # Only unflatten for "+" and "*" operators
    if equation.name in {'f_add', 'f_mul'}:
        if len(equation.children) > 2:
            # Start with the first child and iterate through the rest
            current_node = TreeNode(equation.name, [equation.children[0]])
            for child in equation.children[1:]:
                if isinstance(child, TreeNode) and child.name == equation.name:
                    # If the child has the same operator, extend the current node's children
                    current_node.children.extend(child.children)
                else:
                    # Otherwise, add it as a separate branch
                    current_node.children.append(child)
            # Recursively unflatten the nodes that need it
            for i in range(len(current_node.children)):
                current_node.children[i] = unflatten(current_node.children[i])
            return current_node
        else:
            # If there are 2 or fewer children, no need to unflatten
            return equation
    else:
        # If the operator is not "+" or "*", return the equation as is
        return equation
    
def reduce_brack_add(equation):
    number = []
    for i in range(len(equation.children)-1, -1, -1):
        if equation.children[i].name[:2] == "d_":
            number.append(int(equation.children.pop(i).name[2:]))
    #print(number)
    number = tree_form("d_" + str(sum(number)))
    equation.children.append(number)
    return unflatten(equation)
    pass

def integrate_constant(equation):
    equation = equation.children[0]
    equation = flatten_tree(equation)
    constant = None
    for i in range(len(equation.children)-1,-1,-1):
        if equation.children[i].name[:2] == "d_":
            constant = equation.children.pop(i)
            break
    if len(equation.children) == 1:
        equation = equation.children[0]
    elif len(equation.children) > 2:
        equation = unflatten(equation)
    output = TreeNode("f_mul", [constant, TreeNode("f_int", [equation])])
    return output
def arrange_mul(equation):
    list_fx = []
    orig = copy.deepcopy(equation)
    for i in range(len(equation.children)-1,-1,-1):
        if "f_" in str_form(equation.children[i]).replace("f_add", "g_add").replace("f_pow", "g_pow").replace("f_mul", "g_mul"):
            list_fx.append(equation.children.pop(i))

    if len(list_fx) < 2 or len(equation.children) < 2:
        return unflatten(orig)
    
    fx_eq = TreeNode("f_mul", list_fx)
    
    return TreeNode("f_mul", [fx_eq, equation])
def collect(equation):
    output = []
    def collect_helper(equation, prev):
        if equation.name == "f_add" and prev == "f_pow":
            output.append(str_form(equation))
            return
        for child in equation.children:
            if child.name in ["f_add", "f_mul", "f_pow"]:
                collect_helper(child, equation.name)
            else:
                output.append(str_form(child))
    collect_helper(equation, equation.name)
    output = list(set(output))
    output = [x for x in output if tree_form(x).name[:2] != "d_"]
    return output
eq = None

def operate_poly_mul(equation):
    #print(string_equation(str_form(equation)))
    length = len(equation.children[0].children)
    solution = []
    solution.append(TreeNode("f_mul", [equation.children[0].children[0], equation.children[1].children[0]]))
    for i in range(1,length):
        solution.append(TreeNode("f_add", [equation.children[0].children[i], equation.children[1].children[i]]))
    solution = TreeNode("f_poly", solution)
    return copy.deepcopy(solution)

def operate_poly_add(equation):
    length = len(equation.children[0].children)
    if all(str_form(equation.children[0].children[i]) == str_form(equation.children[1].children[i]) for i in range(1,length)):
        #print("HHI")
        tmp = copy.deepcopy(equation)
        tmp.children[0].children[0] = TreeNode("f_add", [tmp.children[0].children[0], tmp.children[1].children[0]])
        return tmp.children[0]
    return None
def poly_add(equation):
    #print(string_equation(str_form(equation)))
    length = len(equation.children[0].children)
    dic_type = {}
    for i in range(len(equation.children)):
        tmp = copy.deepcopy(equation.children[i])
        tmp.children.pop(0)
        if str_form(tmp) in dic_type.keys():
            dic_type[str_form(tmp)] = TreeNode("f_add", [equation.children[i].children[0], dic_type[str_form(tmp)]])
        else:
            dic_type[str_form(tmp)] = equation.children[i].children[0]
    brac = []
    for key in dic_type.keys():
        tmp = tree_form(key)
        tmp.children = [dic_type[key]] + tmp.children
        brac.append(tmp)
    if len(brac) != 1:
        return TreeNode("f_add", brac)
    return brac[0]
def operate_poly_pow(equation):
    length = len(equation.children[0].children)
    if all(str_form(equation.children[1].children[i]) == "d_0" for i in range(1,length)):
        tmp = copy.deepcopy(equation)
        tmp.children[0].children[0] = TreeNode("f_pow", [equation.children[0].children[0], equation.children[1].children[0]])
        for i in range(1, length):
            tmp.children[0].children[i] = TreeNode("f_mul", [equation.children[0].children[i], equation.children[1].children[0]])
        return tmp.children[0]
    return None
def arithmetic(equation):
    tmp = sorted(search(str_form(equation), 1000, [None, None, None]), key=lambda x: len(string_equation(x)))
    if tmp != []:
        return tree_form(tmp[0])
    return equation

def change(equation):
    def change_helper(equation):
        output = None
        if len(equation.children) > 1 and equation.children[0].name == "f_poly" and equation.children[1].name == "f_poly":
            if equation.name == "f_mul":
                output = operate_poly_mul(equation)
            elif equation.name == "f_add":
                output = operate_poly_add(equation)
            elif equation.name == "f_pow":
                output = operate_poly_pow(equation)
        if output:
            return output
        return equation
    for item in [equation]+break_equation(equation):
        equation = replace(equation, item, change_helper(item))
    
    return equation
    
def reduce(eq):
    tmp2 = sorted([y for y in search(eq, 1000, ["formula-list-6/convert_division.txt", None, None]) if "f_div" not in y], key=lambda x: len(string_equation(x)))
    if tmp2 != []:
        eq = tmp2[0]
    term = collect(tree_form(eq))
    zero = []
    for i in range(len(term)+1):
        zero.append(tree_form("d_0"))
    eq = tree_form(eq.replace("d_", "s_"))
    
    zero[0] = tree_form("d_1")
    def find_sub(term):
        for i in range(len(term)):
            for j in range(len(term)):
                if i >= j:
                    continue
                if term[i] in [str_form(x) for x in break_equation(tree_form(term[j]))]:
                    term[i], term[j] = term[j], term[i]
                    return term
        return None
    while True:
        tmp4 = find_sub(term)
        if tmp4 is None:
            break
        term = tmp4
    for i in range(len(term)):
        zero[i+1] = tree_form("d_1")
        eq = replace(eq, tree_form(term[i].replace("d_", "s_")), TreeNode("f_poly", copy.deepcopy(zero)))
        zero[i+1] = tree_form("d_0")

    zero.pop(0)
    for i in range(-100, 100):
        eq = replace(eq, tree_form("s_" + str(i)), TreeNode("f_poly", [tree_form("d_" + str(i))] + copy.deepcopy(zero)))
    #print(string_equation(str_form(eq)))

    def calc_mul(eq):
        while True:
            orig = str_form(eq)
            eq = change(eq)
            eq = arithmetic(eq)
            if orig == str_form(eq):
                break
        return eq
    def calc_add(eq):
        while True:
            orig = str_form(eq)
            for q in break_equation(eq):
                if q.name == "f_add":
                    eq = replace(copy.deepcopy(eq), q, unflatten(poly_add(flatten_tree(q))))
            eq = arithmetic(eq)
            if orig == str_form(eq):
                break
        return eq
    eq = calc_mul(eq)
    eq = calc_add(eq)
    eq = calc_mul(eq)
    #print(str_form(eq))
    
    for i in range(-100, 100):
        eq = replace(eq, TreeNode("f_poly", [tree_form("d_" + str(i))] + copy.deepcopy(zero)),  tree_form("s_" + str(i)))

    for equation in copy.deepcopy(break_equation(eq)):
        if equation.name == "f_poly":
            brac = []
            if equation.children[0].name != "d_1":
                brac.append(equation.children[0])
            for i in range(1,len(equation.children)):
                if equation.children[i].name == "d_0":
                    continue
                elif equation.children[i].name == "d_1":
                    brac.append(tree_form(term[i-1]))
                else:
                    brac.append(TreeNode("f_pow", [tree_form(term[i-1]), copy.deepcopy(equation.children[i])]))
            if len(brac) == 1:
                brac = brac[0]
            elif len(brac) != 2:
                brac = unflatten(TreeNode("f_mul", brac))
            else:
                brac = TreeNode("f_mul", brac)
            eq = replace(eq, equation, brac)
    #print(string_equation(str_form(eq)))
    eq = str_form(eq).replace("s_", "d_")
    tmp2 = sorted(search(eq, 1000, ["formula-list-6/convert_division_2.txt", None, None]), key=lambda x: len(string_equation(x)))
    if tmp2 != []:
        eq = tmp2[0]
    
    return eq
import math_parser_1

def reduce_single_equation(eq_list):
    output = []
    for eq in eq_list:
        print(string_equation(eq))
        eq2 = search(eq, 3, [None, "single_equation_3.txt", None])
        for item in [eq]+eq2:
            #print(string_equation(item))
            tmp = is_linear_equation(tree_form(item))
            
            if tmp is not None:
                #print(string_equation(tmp))
                output.append(tmp)
                break
    orig = copy.deepcopy(output)
    #print(orig)
    for i in range(len(output),0,-1):
        for output in itertools.combinations(orig, i):
            output = list(output)
            all_key = []
            for item in output:
                #print(output)
                all_key += item.keys()
            all_key = list(set(all_key) - set(["const"]))
            matrix = []
            answer = []
            for i in range(len(output)):
                matrix.append([0]*len(all_key))
            for i in range(len(all_key)):
                for j in range(len(output)):
                    if all_key[i] in output[j].keys():
                        matrix[j][i] = float(output[j][all_key[i]][2:])
            for j in range(len(output)):
                answer.append(float(output[j]["const"][2:]))
            #print(matrix)
            #print(answer)
            matrix = np.array(matrix)
            answer = np.array(answer)
            result, residuals, __, ___ = np.linalg.lstsq(matrix, answer, rcond=None)
            
            rank = np.linalg.matrix_rank(matrix)
            
            if rank >= matrix.shape[1]:   
                for i in range(len(result)):
                    for j in range(len(eq_list)):
                        eq_list[j] = str_form(replace(tree_form(eq_list[j]), tree_form(all_key[i]), tree_form("d_"+str(result[i]))))
                for i in range(len(result)):
                    eq_list.append(str_form(TreeNode("f_eq", [tree_form("d_"+str(result[i])), tree_form(all_key[i])])))
                for i in range(len(eq_list)-1,-1,-1):
                    if "v_" not in eq_list[i]:
                        eq_list.pop(i)
                for item in eq_list:
                    print(string_equation(item))
                return copy.deepcopy(eq_list)
    
def collect_var(eq):
    var_list = []
    for item in break_equation(eq):
        if item.name[:2] == "v_":
            var_list.append(str_form(item))
    var_list = list(set(var_list))
    if any(str_form(eq).count(x) != 1 for x in var_list):
        
        return False
    return True
def is_linear_equation(eq):
    if collect_var(eq) == False:
        return None
    
    if eq.name != "f_eq":
        
        return None
    #print(eq.children[1].name[:2])
    if eq.children[1].name[:2] != "d_":
        return None
    #print("JHI")
    eq = flatten_tree(eq)
    #print("HI")
    #print(str_form(eq))
    if eq.children[0].name[:2] == "v_":
        return {eq.children[0].name: "d_1", "const": eq.children[1].name}
    #print(str_form(eq))
    #if eq.children[0].name == "f_mul":
    #    print("JO")
    if eq.children[0].name == "f_mul" and eq.children[0].children[0].name[:2] == "d_" and eq.children[0].children[1].name[:2] == "v_":
        
        return {eq.children[0].children[1].name: eq.children[0].children[0].name, "const": eq.children[1].name}
    
    if eq.children[0].name != "f_add":
        return None
    def is_term(eq):
        if eq.name[:2] == "d_":
            return "const", eq.name
        if eq.name[:2] == "v_":
            return eq.name, "d_1"
        if eq.name != "f_mul":
            return None
        if len(eq.children) != 2:
            return None
        if eq.children[0].name[:2] != "d_":
            return None
        if eq.children[1].name[:2] != "v_":
            return None
        else:
            if str_form(eq).count(eq.children[1].name) != 1:
                return None
        return eq.children[1].name, eq.children[0].name
    output = {"const": eq.children[1].name}
    for child in eq.children[0].children:
        tmp2 = is_term(child)
        if tmp2 is not None:
            output[tmp2[0]] = tmp2[1]
        else:
            return None
    return output
"""
while True:
    tmp = input(">>> ")
    if tmp.split(" ")[0] == "equation":
        eq = math_parser_1.take_input(tmp.split(" ")[1])
    elif tmp.split(" ")[0] == "equation_list":
        eq = [math_parser_1.take_input(x) for x in tmp.split(" ")[1:]]
        
    elif tmp == "solve_linearly":
        if isinstance(eq, list):
            reduce_single_equation(eq)
    elif tmp == "reduce":
        eq = reduce(eq)
        
        eq = tree_form(eq)
        
        for item in collect(eq):
            item = tree_form(item)
            if item.name[:2] == "v_":
                continue
            eq = copy.deepcopy(replace(eq, item.children[0], tree_form(reduce(str_form(item.children[0])))))
                
        eq = str_form(eq)
        print(string_equation(eq))
    elif tmp == "integrate constant":
        eq = tree_form(eq)
        for equation in break_equation(eq):
            if equation.name == "f_int":
                eq = replace(eq, equation, integrate_constant(eq))
        eq = str_form(eq)
        print(string_equation(eq))
    elif tmp == "integrate simple":
        tmp2 = sorted([y for y in search(eq, 1000, [None, "formula-list-8/integrate_simple.txt", None]) if "f_int" not in y], key=lambda x: len(string_equation(x)))
        if tmp2 != []:
            eq = tmp2[0]
            print(string_equation(eq))
        else:
            print("failed")
"""
"""
depth = 10
for _ in range(2):
    eq = sorted(search(eq, depth, ["formula-list-6/high-main.txt", "formula-list-6/medium-main.txt", "formula-list-6/low-main.txt"]), key=lambda x: len(x)+x.count("f_sin")*100)[0]
    depth -= 3
    print("re-centering")
#eq = sorted(search(eq, 10, ["formula-list/high-main.txt", "formula-list/medium-main.txt", "formula-list/low-main.txt"]), key=lambda x: len(x))[0]
print(string_equation(eq))
"""
