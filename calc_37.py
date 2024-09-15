print_on= True
import parser_4
import sys
import tkinter as tk
from tkinter import Text, Scrollbar
import copy
import threading
import time

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
        global all_angles
        
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
            if True: # if only numbers
                if equation.name == "f_xangle" and print_angle_3(equation.children[0].name[2:]+equation.children[1].name[2:]+equation.children[2].name[2:]) in all_angles:
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return tree_form(str_form(equation).replace("f_xangle", "f_angle"))
                        
                elif equation.name == "f_xline":
                    a, b = equation.children[0].name[2:], equation.children[1].name[2:]
                    if (a2n(a),a2n(b)) in point_pairs or (a2n(b),a2n(a)) in point_pairs:
                        count_target_node -= 1
                        if count_target_node == 0:
                            return tree_form(str_form(equation).replace("f_xline", "f_line")) 
                    new = [x for x in all_angles if (x[0] == a and x[2] == b) or (x[0] == b and x[2] == a)]
                    if any(straight_line([a2n(y) for y in x]) for x in new):
                        count_target_node -= 1
                        if count_target_node == 0:
                            return tree_form(str_form(equation).replace("f_xline", "f_line"))
                elif equation.name == "f_xparallel":
                    a = line_fx(line_sort(equation.children[0].children[0].name[2:] + equation.children[0].children[1].name[2:]))
                    b = line_fx(line_sort(equation.children[1].children[0].name[2:] + equation.children[1].children[1].name[2:]))
                    p1 = str_form(TreeNode("f_parallel", [a, b]))
                    p2 = str_form(TreeNode("f_parallel", [b, a]))
                    if p1 in eq_list or p2 in eq_list:
                        count_target_node -= 1
                        if count_target_node == 0:
                            return tree_form("d_true")
                elif equation.name == "f_exist" and "f_x" not in str_form(equation.children[0]): # power should be two or a natural number more than two
                    count_target_node -= 1
                    if count_target_node == 0:
                        return tree_form("d_true")
                elif equation.name == "f_eq" and "f_x" not in str_form(equation):
                    if "angle" in str_form(equation):
                        a = print_angle_3(equation.children[0].children[0].name[2:] + equation.children[0].children[1].name[2:] + equation.children[0].children[2].name[2:])
                        b = print_angle_3(equation.children[1].children[0].name[2:] + equation.children[1].children[1].name[2:] + equation.children[1].children[2].name[2:])
                        try:
                            row = [0]*len(all_angles)
                            row[all_angles.index(a)] = 1
                            row[all_angles.index(b)] = -1
                            row_2 = [0]*len(all_angles)
                            row_2[all_angles.index(a)] = -1
                            row_2[all_angles.index(b)] = 1
                            if (row in convert_to_int(matrix) and convert_to_int(matrix_eq)[convert_to_int(matrix).index(row)] == 0) or\
                               (row_2 in convert_to_int(matrix) and convert_to_int(matrix_eq)[convert_to_int(matrix).index(row_2)] == 0):
                                count_target_node -= 1
                                if count_target_node == 0:
                                    return tree_form("d_true")
                        except:
                            pass
                    else:
                        a = line_sort(equation.children[0].children[0].name[2:] + equation.children[0].children[1].name[2:])
                        b = line_sort(equation.children[1].children[0].name[2:] + equation.children[1].children[1].name[2:])
                        a = index_line(*a)
                        b = index_line(*b)
                        row = [0]*len(line_counter)
                        row[a] = 1
                        row[b] = -1
                        row_2 = [0]*len(line_counter)
                        row_2[a] = -1
                        row_2[b] = 1
                        fix_line_matrix()
                        if row in convert_to_int(line_matrix) or row_2 in convert_to_int(line_matrix):
                            count_target_node -= 1
                            if count_target_node == 0:
                                return tree_form("d_true")
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
def flatten_tree(node):
    if not node.children:
        return node
    if node.name in {"f_add", "f_mul"}: # commutative property supporting functions
        merged_children = [] # merge all the children
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
def convert_to_int(matrix):
    # Check if the matrix is 1D or 2D
    if isinstance(matrix[0], list):
        # If it's 2D, convert each element in each row
        return [[int(element) for element in row] for row in matrix]
    else:
        # If it's 1D, convert each element directly
        return [int(element) for element in matrix]
def convert_angle_straight(angle_list, val):
    if len(angle_list) == 1:
        return str_form(TreeNode("f_eq", [convert_angle(angle_list[0]), tree_form("d_"+str(val))]))
    else:
        coll = TreeNode("f_add", [])
        coll.children.append(convert_angle(angle_list.pop(-1)))
        coll.children.append(convert_angle(angle_list.pop(-1)))
        while len(angle_list)> 0:
            coll = copy.deepcopy(TreeNode("f_add", [coll, convert_angle(angle_list.pop(-1))]))
        return str_form(TreeNode("f_eq", [coll, tree_form("d_"+str(val))]))

def line_fx(line_input):
    a = line_input[0]
    b = line_input[1]
    return TreeNode("f_line", [tree_form("d_"+a), tree_form("d_"+b)])
line_counter = []
def index_line(a, b):
    global line_counter
    global point_pairs
    a, b = line_sort(a+b)
    a, b = a2n(a), a2n(b)
    if (a,b) in line_counter:
        return line_counter.index((a,b))
    else:
        line_counter.append((a,b))
        fix_line_matrix()
        return len(line_counter) - 1
def index_line_matrix(index):
    global line_counter
    a, b = line_counter[index]
    return line_sort(n2a(a)+n2a(b))
#def all_int(lst):
    
def convert_line(line1, line2, line_add):
    return str_form(TreeNode("f_eq", [TreeNode("f_add", [line_fx(line1), line_fx(line2)]), line_fx(line_add)]))
def convert_line_eq(line1, line2):
    return str_form(TreeNode("f_eq", [line_fx(line1), line_fx(line2)]))
def convert_angle_eq(angle1, angle2):
    return str_form(TreeNode("f_eq", [convert_angle(angle1), convert_angle(angle2)]))

def convert_angle_add(angle1, angle2, angle_add):
    return str_form(TreeNode("f_eq", [TreeNode("f_add", [convert_angle(angle1), convert_angle(angle2)]), convert_angle(angle_add)]))

def convert_angle_2(angle):
    return TreeNode("f_angle", [tree_form("d_"+angle[0]), tree_form("d_"+angle[1]), tree_form("d_"+angle[2])])

def convert_angle(angle):
    if angle[0] == "(":
        angle = angle.replace("(360-", "").replace(")", "")
        angle = print_angle_3(angle)
        return TreeNode("f_sub", [tree_form("d_360"), TreeNode("f_angle", [tree_form("d_"+angle[0]), tree_form("d_"+angle[1]), tree_form("d_"+angle[2])])])
    angle = print_angle_3(angle)
    return TreeNode("f_angle", [tree_form("d_"+angle[0]), tree_form("d_"+angle[1]), tree_form("d_"+angle[2])])

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
    transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), None, None, True) # perform arithmetic
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to read formula file
def return_formula_file(file_name):
    with open(file_name, 'r') as file:
      content = file.read()
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] # alternative formula lhs and then formula rhs
    output_f = [x[i] for i in range(1, len(x), 2)]
    #for i in range(len(input_f)):
    #    print(string_equation(input_f[i]) + " = " + string_equation(output_f[i]))
    input_f = [tree_form(item) for item in input_f] # convert into tree form
    output_f = [tree_form(item) for item in output_f]
    
    return [input_f, output_f] # return

def search(equation, depth, file_list, auto_arithmetic=True, visited=None):
    if depth == 0: # limit the search
        return None
    if visited is None:
        visited = set()
    equation = fix_angle_line(equation)
    print_text(string_equation(equation))
    if equation in visited:
        return None
    visited.add(equation)
    output =[]
    if file_list[0]:
      output += generate_transformation(equation, file_list[0])
    if auto_arithmetic:
      output += generate_arithmetical_transformation(equation)
    if len(output) > 0:
      if depth == 1:
        depth += 1
      output = [output[0]]
    else:
      if file_list[1]:
        output += generate_transformation(equation, file_list[1])
      if not auto_arithmetic:
        output += generate_arithmetical_tran
        sformation(equation)
      if file_list[2] and len(output) == 0:
          output += generate_transformation(equation, file_list[2])
    for i in range(len(output)):
        result = search(output[i], depth-1, file_list, auto_arithmetic, visited) # recursively find even more equals
        if result is not None:
            output += result # hoard them
    output = list(set(output))
    return output

# fancy print
def string_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name # leaf node
    s = "(" # bracket
    if len(equation_tree.children) == 1 or equation_tree.name in ["f_if", "f_xcongruent", "f_congruent", "f_triangle", "f_xangle", "f_xline", "f_angle", "f_line", "f_parallel"]:
        s = equation_tree.name[2:] + s
    sign = {"f_if": ",", "f_xparallel": ",", "f_xcongruent": ",", "f_congruent": ",", "f_triangle": "?", "f_add": "+", "f_and": "^", "f_dif": "?", "f_mul": "*", "f_eq": "=", "f_sub": "-", "f_angle": "?", "f_xangle": "?", "f_parallel": ",", "f_xline": "?", "f_exist": "?", "f_line": "?"} # operation symbols
    for child in equation_tree.children:
        s+= string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
    s = s[:-1] + ")"
    return s

# fancy print main function
def string_equation(eq):
    eq = eq.replace("d_any1", "#")
    eq = eq.replace("d_any2", "#")
    eq = eq.replace("d_", "")
    
    return string_equation_helper(tree_form(eq)).replace("?", "")

def replace(equation, find, r):
  if str_form(equation) == str_form(find):
    return r
  col = TreeNode(equation.name, [])
  for child in equation.children:
    col.children.append(replace(child, find, r))
  return col

def line_eq(line1, line2):
    if line1 == line2:
        return True
    line1 = a2n(line1[0]), a2n(line1[1])
    line2 = a2n(line2[0]), a2n(line2[1])
    row = [0]*len(line_counter)
    row[line_counter.index(line1)] = 1
    row[line_counter.index(line2)] = -1
    if row in convert_to_int(line_matrix):
        return True
    row[line_counter.index(line1)] = -1
    row[line_counter.index(line2)] = 1
    if row in convert_to_int(line_matrix):
        return True
    return False
def angle_eq(angle1, angle2):
    if angle1 == angle2:
        return True
    row = [0]*len(all_angles)
    row[all_angles.index(angle1)] = 1
    row[all_angles.index(angle2)] = -1
    if row in convert_to_int(matrix) and int(matrix_eq[convert_to_int(matrix).index(row)]) == 0:
        return True
    row[all_angles.index(angle1)] = -1
    row[all_angles.index(angle2)] = 1
    if row in convert_to_int(matrix) and int(matrix_eq[convert_to_int(matrix).index(row)]) == 0:
        return True
    return False

def line_counter_convert():
    output = []
    for item in line_counter:
        output.append(line_sort(n2a(item[0])+n2a(item[1])))
    return output

def sss_rule(a1,a2,a3, b1,b2,b3):
    line = [line_sort(a1+a2), line_sort(b1+b2), line_sort(a2+a3), line_sort(b2+b3), line_sort(a1+a3), line_sort(b1+b3)]
    #print(line)
    #print(line_counter)
    for item in line:
        if item not in line_counter_convert():
            return False
        
    return line_eq(line[0], line[1]) and line_eq(line[2], line[3]) and line_eq(line[4], line[5])
def sas_rule(a1,a2,a3, b1,b2,b3):
    line = [line_sort(a1+a2), line_sort(b1+b2), line_sort(a2+a3), line_sort(b2+b3)]
    angle = [print_angle_3(a1+a2+a3), print_angle_3(b1+b2+b3)]

    if set(a1+a2+a3+b1+b2+b3) == set("CDABEC") and line_eq(line[0], line[1]):
        print("hi3" + angle[0] + " " + angle[1] + " " + line[2] + " " + line[3])
    
    for item in line:
        if item not in line_counter_convert():
            return False
    for item in angle:
        if item not in all_angles:
            #print(angle)
            return False
    
    return line_eq(line[0], line[1]) and angle_eq(angle[0], angle[1]) and line_eq(line[2], line[3])
def aas_rule(a1,a2,a3, b1,b2,b3):
    line = [line_sort(a1+a2), line_sort(b1+b2), line_sort(a1+a3), line_sort(b1+b3)]
    angle = [print_angle_3(a1+a2+a3), print_angle_3(b1+b2+b3), print_angle_3(a3+a1+a2), print_angle_3(b3+b1+b2)]
    for item in line:
        if item not in line_counter_convert():
            return False
    for item in angle:
        if angle not in all_angles:
            return False
    return line_eq(line[0], line[1]) and angle_eq(angle[0], angle[1]) and angle_eq(line[2], line[3])
def proof_fx_3(angle1, angle2):
    global eq_list
    #print(angle1, angle2)
    angle_1 = TreeNode("f_triangle", [tree_form("d_"+angle1[0]), tree_form("d_"+angle1[1]), tree_form("d_"+angle1[2])])
    angle_2 = TreeNode("f_triangle", [tree_form("d_"+angle2[0]), tree_form("d_"+angle2[1]), tree_form("d_"+angle2[2])])
    eq = TreeNode("f_congruent", [angle_1, angle_2])
    eq = str_form(eq)
    
    for angle in [angle1+angle2, angle2+angle1]:
        if sss_rule(*angle) or sas_rule(*angle) or aas_rule(*angle):
            eq_list.append(fix_angle_line(eq))
            do_cpct()
            return eq
    return None

def proof_fx(eq):
    global eq_list
    output = search(eq, 7, [None, "formula-list-9/h.txt", None])
    for item in output:
        if item == eq:
            continue
        output2 = search(item, 7, [None, "formula-list-9/b.txt", None])
        if "d_true" in output2:
            eq_list.append(fix_angle_line(eq))
            do_cpct()
            return eq
    return None

def proof_fx_2(eq):
    global eq_list
    global matrix
    global matrix_eq
    #final
    output = search(eq, 7, [None, "formula-list-9/j.txt", None])
    for item in output:
        if "f_parallel" not in item and "f_if" not in item:
            item = convert_form(item)
            if eq:
                m, meq = find_all(item, len(all_angles))
                matrix.append(m)
                matrix_eq.append(meq)
    return None

import math
import numpy as np
def find_circle_center_radius(points):
    # This function is predefined and kept unchanged
    return (150, 200), 10

def find_intersections(lines, circle_center, radius):
    h, k = circle_center
    r = radius
    intersections = []

    def line_circle_intersections(x1, y1, x2, y2):
        # Calculate line coefficients
        dx = x2 - x1
        dy = y2 - y1
        A = dx**2 + dy**2
        B = 2 * (dx * (x1 - h) + dy * (y1 - k))
        C = (x1 - h)**2 + (y1 - k)**2 - r**2
        
        discriminant = B**2 - 4 * A * C
        
        if discriminant < 0:
            return []  # No intersection

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-B + sqrt_discriminant) / (2 * A)
        t2 = (-B - sqrt_discriminant) / (2 * A)

        points = []
        for t in (t1, t2):
            if 0 <= t <= 1:
                ix = x1 + t * dx
                iy = y1 + t * dy
                points.append((ix, iy))
        
        return points
    final = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        
        if (x1, y1) == circle_center:
            x3, y3 = x2, y2
        elif (x2, y2) == circle_center:
            x3, y3 = x1, y1
        else:
            continue
        final.append((x3, y3))
        intersections.extend(line_circle_intersections(x1, y1, x2, y2))

    return list(zip(intersections, final))

def sort_points_anticlockwise(points, center):
    h, k = center

    def angle_from_center(point):
        point = point[0]
        return math.atan2(point[1] - k, point[0] - h)
    
    return sorted(points, key=angle_from_center)

from PIL import Image, ImageDraw, ImageFont, ImageTk

def draw_points_and_lines(points, lines, image_size=(1000, 1000), point_radius=5, point_color=(0, 0, 0), line_color=(255, 0, 0)):
    # Create a white image
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)

    # Draw points
    for x, y in points:
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)

    # Draw lines
    for (x1, y1), (x2, y2) in lines:
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=2)

    return image
def is_number(s):
    try:
        float(s)  # Convert to float
        return True
    except ValueError:
        return False
def compare_trees(tree1, tree2):
    # If both nodes are integers or floats, compare them as floats
    if tree1.name[:2] == "d_" and tree2.name[:2] == "d_" and is_number(tree1.name[2:]) and is_number(tree2.name[2:]):
        return abs(float(tree1.name[2:])-float(tree2.name[2:])) < 0.001
    
    # If one of the nodes is not a number, or the names are not equal, return False
    if tree1.name != tree2.name:
        return False
    
    # If the number of children is not the same, return False
    if len(tree1.children) != len(tree2.children):
        return False
    
    # Recursively compare all children of both trees
    for child1, child2 in zip(tree1.children, tree2.children):
        if not compare_trees(child1, child2):
            return False
    
    # If all checks passed, the trees are equal
    return True
def remove_duplicate_trees(tree_list):
    unique_trees = []
    
    for tree in tree_list:
        # Check if the current tree is already in unique_trees using compare_trees
        is_duplicate = any(compare_trees(tree, unique_tree) for unique_tree in unique_trees)
        
        # If no duplicate is found, add the tree to the unique list
        if not is_duplicate:
            unique_trees.append(tree)
    
    return unique_trees
def line_intersection(p1, p2, q1, q2, epsilon=0.1):
    """Find the intersection point of two line segments (p1p2 and q1q2), handling precision issues."""

    def is_near_zero(x):
        """Check if a value is close to zero."""
        return abs(x) < epsilon

    def is_between(a, b, c):
        """Check if point b is between points a and c."""
        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

    A, B, C, D = p1, p2, q1, q2

    def line_params(p1, p2):
        """Calculate line parameters A, B, C for the line equation Ax + By = C."""
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = A * p1[0] + B * p1[1]
        return A, B, C

    # Line 1 parameters
    A1, B1, C1 = line_params(A, B)
    # Line 2 parameters
    A2, B2, C2 = line_params(C, D)

    # Calculate the determinant
    det = A1 * B2 - A2 * B1

    if is_near_zero(det):
        return None  # Lines are parallel or coincident

    # Calculate intersection point
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    intersect = (x, y)

    # Ensure the intersection point is within both line segments
    if is_between(A, intersect, B) and is_between(C, intersect, D):
        return intersect

    return None




def round_point(point, precision=1):
    """Round a point to a given precision."""
    return (round(point[0] / precision) * precision, round(point[1] / precision) * precision)

def find_intersections_2(points, point_pairs):
    """Find all unique intersection points of the line segments not in the points list."""
    intersections = set()
    
    for i in range(len(point_pairs)):
        p1 = points[point_pairs[i][0]]
        p2 = points[point_pairs[i][1]]
        for j in range(len(point_pairs)):
            if i != j:
                q1 = points[point_pairs[j][0]]
                q2 = points[point_pairs[j][1]]
                if p1 != q1 and p1 != q2 and p2 != q1 and p2 != q2:
                    intersect = line_intersection(p1, p2, q1, q2)
                    if intersect:
                        rounded_intersect = round_point(intersect)
                        intersections.add(rounded_intersect)

    #print(intersections)
    # Filter out intersections that are already in the points list
    filtered_intersections = [point for point in intersections if round_point(point) not in map(round_point, points)]
    
    return filtered_intersections


def a2n(letter):
    return ord(letter) - ord("A")
def a2n2(line):
    return (a2n(line[0]), a2n(line[1]))
#points = [(100, 300), (400, 300), (275, 120), (275,300)]
#point_pairs = [(0, 3), (1, 3), (2,3), (2,0), (2,3)]  # Connect the triangle vertices

#points = [(450, 250), (350, 423), (150, 423), (50, 250), (150, 77), (350, 77)]
#point_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 1), (1, 2), (0, 2), (0, 2), (2, 3), (0, 3), (0, 3), (3, 4), (0, 4), (0, 4), (4, 5), (0, 5)]

#points = [(100, 100), (100, 150), (100, 200), (50, 150), (150, 150)]
#point_pairs = [a2n2("AB"), a2n2("BC"), a2n2("BE"), a2n2("BD"), a2n2("AD"), a2n2("CE")]

points = []
point_pairs = []

buffer_eq_list = []
eq_list = []  
# Generate lines from point pairs
lines = []
matrix = []
matrix_eq = []
line_matrix = []
import itertools
def surrounding_angle(point_name):
    global lines
    global points
    global point_pairs
    global buffer_eq_list
    curr = points[point_name]
    radius = 20

    # Find intersections of lines with the circle
    intersections = find_intersections(lines, curr, radius)
    #print(intersections)
    # Sort intersections in anti-clockwise order
    sorted_intersections = sort_points_anticlockwise(intersections, curr)

    return [points.index(x[1]) for x in sorted_intersections]
def calculate_triangle_area(point_list):
    (x1, y1), (x2, y2), (x3, y3) = point_list
    area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2
    return area
def n2a(number):
    return chr(number + ord('A'))
def straight_line(point_list):
    global lines
    global points
    global point_pairs
    tmp = calculate_triangle_area([points[x] for x in point_list])
    #print(tmp)
    return tmp < 100
def draw_points_and_lines(points, lines, image_size=(1000, 1000), point_radius=5, point_color=(0, 0, 0), line_color=(255, 0, 0), text_color=(0, 0, 0)):
    # Create a white image
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Optionally, load a font for text (adjust the path and size as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 32)  # You can replace "arial.ttf" with the path to any TTF font file
    except IOError:
        font = ImageFont.load_default()

    # Draw points and annotate them
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)
        # Draw the point number
        draw.text((x + point_radius + 5, y - point_radius - 5), n2a(index), fill=text_color, font=font)

    # Draw lines
    for (x1, y1), (x2, y2) in lines:
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=2)

    return image

def print_text(text, color="black", force=False, auto_next_line=True):
    global print_on
    if not force and not print_on:
        return
    """This function will display text in the text widget."""
    if auto_next_line:
        console.insert(tk.END, text + "\n", color)
    else:
        console.insert(tk.END, text, color)
    console.see(tk.END)  # Scroll to the end

def display_image(image_path):
    """This function will display an image in the image label."""
    img = Image.open(image_path)
    img = img.resize((250, 250))  # Resize image to fit the display area
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk 


def print_diagram():
    global lines
    global points
    global point_pairs
    
    # Draw points and lines on the image
    image = draw_points_and_lines(points, lines)
    
    # Save image to file
    image.save('points_and_lines_image.png')
    
    display_image('points_and_lines_image.png')

  
#surrounding_angle(0)
def travel_till_end(start, step):
    global lines
    global points
    global point_pairs
    done = False
    step_taken = [step]
    while not done:
        done = True
        for item in surrounding_angle(step):
            if straight_line([step, start, item]) and item not in step_taken and start != item and step != start and step != item:
                step_taken.append(item)
                step = item
                #print(n2a(step))       
                done = False
                break
    return step

def sur(angle):
    global lines
    global points
    global point_pairs
    count = 0
    if a2n(angle[0]) in surrounding_angle(a2n(angle[1])):
        count += 1
    if a2n(angle[2]) in surrounding_angle(a2n(angle[1])):
        count += 1
    return count
           

def print_angle(a, b, c, a_do=True, c_do=True):
    global lines
    global points
    global point_pairs
    #print(n2a(a), n2a(b), n2a(c))
    if a_do:
        a = travel_till_end(b, a)
    else:
        a = travel_till_end(b, a)
        a = travel_till_end(b, a)
    if c_do:
        c = travel_till_end(b, c)
    else:
        c = travel_till_end(b, c)
        c = travel_till_end(b, c)
    #print(n2a(a), n2a(b), n2a(c))
    m, n = sorted([a, c])
    return n2a(m) + n2a(b) + n2a(n)
def print_angle_2(angle, a_do=True, c_do=True):
    global lines
    global points
    global point_pairs
    x = angle
    return print_angle(a2n(x[0]), a2n(x[1]), a2n(x[2]), a_do, c_do)
def print_angle_3(angle):
    lst = [print_angle_2(angle, True, True), print_angle_2(angle, True, False), print_angle_2(angle, False, True), print_angle_2(angle, False, False)]
    return sorted(lst, key=lambda x: sur(x))[0]
def print_angle_4(a, b, c):
    return print_angle_3(n2a(a)+n2a(b)+n2a(c))
#combine
def combine(a, b):
    global lines
    global points
    global point_pairs
    
    a = print_angle_3(a)
    b = print_angle_3(b)
    if a[1] != b[1]:
        return None
    if len(set(a+b)) != 4:
        return None
    r = a[0] + a[2] + b[0] + b[2]
    r = r.replace([x for x in r if r.count(x) == 2][0], "")
    out = print_angle_3(r[0] + b[1] + r[1])
    #out = print_angle_2(out)
    return out
def calculate_angle(angle):
    global lines
    global points
    global point_pairs
    A = points[a2n(angle[0])]
    O = points[a2n(angle[1])]
    B = points[a2n(angle[2])]
    x_A, y_A = A
    x_O, y_O = O
    x_B, y_B = B
    vector_OA = (x_A - x_O, y_A - y_O)
    vector_OB = (x_B - x_O, y_B - y_O)
    dot_product = vector_OA[0] * vector_OB[0] + vector_OA[1] * vector_OB[1]
    magnitude_OA = math.sqrt(vector_OA[0]**2 + vector_OA[1]**2)
    magnitude_OB = math.sqrt(vector_OB[0]**2 + vector_OB[1]**2)
    cos_theta = dot_product / (magnitude_OA * magnitude_OB)
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees
def angle_sort(angle):
    if a2n(angle[0]) > a2n(angle[2]):
        angle = angle[2] + angle[1] + angle[0]
    return angle
def line_sort(line):
    if a2n(line[0]) > a2n(line[1]):
        line = line[1] + line[0]
    return line
all_angles = []
import math_1
import math_2
def break_equation(equation):
    sub_equation_list = [equation]
    equation = equation
    for child in equation.children: # breaking equation by accessing children
        sub_equation_list += break_equation(child) # collect broken equations
    return sub_equation_list
def find_constant(equation, target_var):
    equation = do_zero_rhs(equation)
    equation = str_form(TreeNode("f_dif", [tree_form(equation)]))
    for i in range(30):
        if i == target_var:
            continue
        equation = str_form(replace(tree_form(equation), tree_form("v_"+str(i)), tree_form("v_30")))
    equation = str_form(replace(tree_form(equation), tree_form("v_"+str(target_var)), tree_form("v_0")))
    output = math_1.search(equation, 10, ["single_equation_3.txt", None, None])
    return round(float(sorted([x for x in output if "v_" not in x], key=lambda x: len(x))[0][2:]))
def do_zero_rhs(equation):
    output = math_1.search(equation, 5, ["single_equation_5.txt", None, None])
    for item in [equation] + output:
        if tree_form(item).name == "f_eq" and tree_form(item).children[1].name == "d_0":
            return str_form(tree_form(item).children[0])
def find_last(equation):
    equation = do_zero_rhs(equation)
    for i in range(30):
        equation = str_form(replace(tree_form(equation), tree_form("v_"+str(i)), tree_form("d_0")))
    output = math_1.search(equation, 5, [None, None, None])
    return round(float(sorted([x for x in output if "v_" not in x], key=lambda x: len(x))[0][2:]))

def find_all(equation, total_var):
    row = []
    for i in range(total_var):
        row.append(find_constant(copy.deepcopy(equation),i))
    tmp = find_last(equation)
    return [row, tmp]

def row_swap(matrix, row1, row2):
    """ Swap two rows in the matrix. """
    matrix[[row1, row2]] = matrix[[row2, row1]]

def row_scale(matrix, row, scale):
    """ Scale (multiply) a row by a non-zero scalar. """
    matrix[row] *= scale

def row_addition(matrix, source_row, target_row, scale):
    """ Add a multiple of one row to another row. """
    matrix[target_row] += scale * matrix[source_row]

def gauss_jordan_elimination(matrix):
    """ Perform Gauss-Jordan elimination to transform the matrix into RREF. """
    rows, cols = matrix.shape
    row = 0
    
    for col in range(cols - 1):  # Ignore the last column (augmented part)
        # Find the pivot row
        pivot_row = np.argmax(np.abs(matrix[row:, col])) + row
        
        if matrix[pivot_row, col] == 0:
            continue  # Skip this column if the pivot is zero
        
        # Swap the current row with the pivot row
        if pivot_row != row:
            row_swap(matrix, row, pivot_row)
        
        # Scale the pivot row
        row_scale(matrix, row, 1 / matrix[row, col])
        
        # Eliminate the current column entries in other rows
        for r in range(rows):
            if r != row:
                row_addition(matrix, row, r, -matrix[r, col])
        
        row += 1
        if row >= rows:
            break
    
    return matrix
def matrix_to_list(matrix):
    # Convert NumPy matrix to a nested list with rounded integers
    if isinstance(matrix, np.ndarray):
        if matrix.ndim > 1:
            # Recursively convert each sub-array
            return [matrix_to_list(sub_matrix) for sub_matrix in matrix]
        else:
            # Convert to a list and round each element
            return [round(value) for value in matrix]
    else:
        # If it's not a NumPy array, assume it's already a list
        return [round(value) for value in matrix]
def detect_inconsistency(matrix):
    matrix = gauss_jordan_elimination(matrix)
    
    for item in matrix_to_list(matrix):
        if all(float(x)<0.01 for x in item[:-1]) and not(float(item[-1]) < 0.01):
            return False
    return True
def unaugment_matrix(augmented_matrix):
    num_cols = augmented_matrix.shape[1] - 1
    A = augmented_matrix[:, :num_cols]
    B = augmented_matrix[:, num_cols:]
    return A, B
def fix_matrix():
    global matrix
    for i in range(len(matrix)):
        if len(matrix[i]) < len(all_angles):
            matrix[i] += [0]*(len(all_angles)-len(matrix[i]))
def fix_line_matrix():
    global line_matrix
    global line_counter
    global point_pairs
    
    # Target size is len(point_pairs) + len(line_counter)
    target_size = len(line_counter)
    
    for i in range(len(line_matrix)):
        # If the row is shorter, add zeros at the end
        if len(line_matrix[i]) < target_size:
            line_matrix[i] += [0] * (target_size - len(line_matrix[i]))
        # If the row is longer, truncate it to the target size
        elif len(line_matrix[i]) > target_size:
            line_matrix[i] = line_matrix[i][:target_size]
def try_line_matrix():
    global line_matrix
    if line_matrix == []:
        return
    A = np.array(line_matrix, dtype=float)
    before_rank = np.linalg.matrix_rank(A)
    new_matrix = gauss_jordan_elimination(A)
    for item in itertools.combinations(line_counter_convert(), 2):
        row = [0]*(len(line_counter)+2)
        row[index_line(*item[0])] = 1
        row[index_line(*item[1])] = -1
        line_matrix.append(row)
        fix_line_matrix()
        if before_rank == np.linalg.matrix_rank(line_matrix):
            pass
        else:
            line_matrix.pop(-1)
def try_matrix():
    global matrix
    global matrix_eq
    global eq_list
    global all_angles
    if matrix == []:
        return

    A = np.array(matrix, dtype=float)
    B = np.array(matrix_eq, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    augmented_matrix = np.hstack((A, B))
    before_rank = np.linalg.matrix_rank(augmented_matrix)
    #print(augmented_matrix)
    new_matrix = gauss_jordan_elimination(augmented_matrix)
    #A = matrix_to_l)
    
    #print(new_matrix)
    #print(all_angles)
    for item in itertools.combinations(all_angles, 2):
        row = [0]*len(matrix[0])
        row[all_angles.index(item[0])] = 1
        row[all_angles.index(item[1])] = -1
        matrix.append(row)
        matrix_eq.append(0)
        if detect_inconsistency(np.hstack((np.array(matrix, dtype=float),np.array(matrix_eq, dtype=float).reshape(-1,1)))) and\
           before_rank == np.linalg.matrix_rank(np.hstack((np.array(matrix, dtype=float),np.array(matrix_eq, dtype=float).reshape(-1,1)))):
            pass
        else:
            matrix.pop(-1)
            matrix_eq.pop(-1)
    for item in matrix_to_list(new_matrix):
        if item[:-1].count(1) == 1 and item[:-1].count(0)+2 == len(item):
            matrix.append(item[:-1])
            matrix_eq.append(item[-1])
            #eq_list.append(convert_angle_straight([all_angles[item[:-1].index(1)]], item[-1]))
def line_matrix_print(print_it=True):
    def remove_duplicate_rows(matrix):
        # Convert each row to a tuple and store in a set to remove duplicates
        unique_rows = set(tuple(row) for row in matrix)
        # Convert tuples back to lists and return the result
        return [list(row) for row in unique_rows]
    
    global line_matrix
    global point_pairs
    fix_line_matrix()
    #print(line_matrix)
    line_matrix = remove_duplicate_rows(line_matrix)
        
    string = "$"
    for i in range(len(line_matrix)):
        for j in range(len(line_matrix[i])):
            if not(abs(line_matrix[i][j]) < 0.01):
                if line_matrix[i][j] < 0:
                    string += "-line("+index_line_matrix(j)+")"
                else:
                    string += "+line("+index_line_matrix(j)+")"
        string += "=0\n"
    string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
    if print_it and string != "":
        print_text(string, "black", False, False)
def matrix_print(print_it=True):
    def remove_duplicates(matrix_2d, array_1d):
        unique_rows = {}
        for row, val in zip(matrix_2d, array_1d):
            row_tuple = tuple(row)
            if row_tuple not in unique_rows:
                unique_rows[row_tuple] = val
        new_matrix_2d = list(unique_rows.keys())
        new_array_1d = list(unique_rows.values())
        return new_matrix_2d, new_array_1d
    global matrix
    global matrix_eq
    global all_angles
    fix_matrix()
    matrix, matrix_eq = remove_duplicates(matrix, matrix_eq)
    for i in range(len(matrix)):
        matrix[i] = list(matrix[i])
    #print(matrix)
    #print(matrix_eq)
    string = "$"
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if not(abs(matrix[i][j]) < 0.01):
                if matrix[i][j] < 0:
                    string += "-angle("+all_angles[j]+")"
                else:
                    string += "+angle("+all_angles[j]+")"
        string += "=" + str(int(matrix_eq[i])) + "\n"
    string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
    if print_it and string != "":
        print_text(string, "black", False, False)
                

def process(print_it=True):
    global lines
    global points
    global point_pairs
    global all_angles
    global fx_call
    global eq_list
    global buffer_eq_list
    global matrix
    global matrix_eq
    global line_matrix
    buffer_eq_list = []
    
    #print("I am prcoess")
    lines = [(points[start], points[end]) for start, end in point_pairs]
    print_diagram() 
    output = []
    find = find_intersections_2(points, point_pairs)
    #if find != []:
    #    return None
    for i in range(len(points)):
        for angle in itertools.combinations(surrounding_angle(i), 2):
            if angle[0] != angle[1]:
                output.append(print_angle_4(angle[0], i, angle[1]))
    #print(output)
    output = list(set(output))
    append_angles = set(output) - set(all_angles)
    reject_angles = set(all_angles) - set(output)
    #print(reject_angles)
    all_angles = all_angles + list(append_angles)

    index_iter = sorted([all_angles.index(item) for item in reject_angles])[::-1]
    #print(index_iter)
    #print(matrix_eq)
    #print(all_angles)
    for i in range(len(matrix)-1,-1,-1):
        for item in index_iter:
            if int(matrix[i][item]) == 1:
                matrix.pop(i)
                break
    for item in index_iter:
        for i in range(len(matrix)):
            matrix[i].pop(item)
        #matrix_eq.pop(item)
        all_angles.pop(item)
    for item in point_pairs:
        s = n2a(item[0])+n2a(item[1])
        index_line(*s)
    #print(all_angles)
    output = []
    
    for i in range(len(points)):
        for angle in all_angles:
            if straight_line([a2n(x) for x in angle]):
                #print(angle)
                output.append(angle)
    output = list(set(output))
    for x in output:
        #buffer_eq_list.append(convert_angle_straight([x], 180))
        #print(string_equation()
        #print(x + " = 180")
        
        index_line(*line_sort(x[0]+x[1]))
        matrix.append([0]*len(all_angles))
        matrix[-1][all_angles.index(x)] = 1
        matrix_eq.append(180)
        
        row = [0]*(len(line_counter)+3)
        row[index_line(x[0], x[1])] = 1
        row[index_line(x[1], x[2])] = 1
        row[index_line(x[2], x[0])] = -1
        line_matrix.append(row)
        #buffer_eq_list.append(convert_line(line_sort(x[0]+x[1]), line_sort(x[1]+x[2]), line_sort(x[0]+x[2])))
        #print(string_equation()
        #print( + " + " +  + " = " + )

    output = []
    for angle in itertools.permutations(all_angles, 3):
        if combine(angle[0], angle[1]) == angle[2]:
            if calculate_angle(angle[2]) > calculate_angle(angle[0]) and\
               calculate_angle(angle[2]) > calculate_angle(angle[1]) and\
               abs(calculate_angle(angle[2]) - calculate_angle(angle[0]) - calculate_angle(angle[1])) < 10:
                if angle[1] + " + " + angle[0] + " = " + angle[2] not in output:
                    matrix.append([0]*len(all_angles))
                    matrix[-1][all_angles.index(angle[0])] = 1
                    matrix[-1][all_angles.index(angle[1])] = 1
                    matrix[-1][all_angles.index(angle[2])] = -1
                    matrix_eq.append(0)
                    output.append(angle[0] + " + " + angle[1] + " = " + angle[2])
                    #buffer_eq_list.append(convert_angle_add(angle[0], angle[1], angle[2]))
                    
    output = list(set(output))
    for angle in itertools.combinations(all_angles, 2):
        if angle[0][1] == angle[1][1] and straight_line([a2n(x) for x in angle[0]]) and straight_line([a2n(x) for x in angle[1]]):
            tmp1 = angle_sort(angle[1][0] + angle[0][1] + angle[0][2])
            tmp2 = angle_sort(angle[0][0] + angle[1][1] + angle[1][2])
            output.append(tmp1 + " = " + tmp2)
            #buffer_eq_list.append(convert_angle_eq(tmp1, tmp2))
            matrix.append([0]*len(all_angles))
            matrix[-1][all_angles.index(tmp1)] = 1
            matrix[-1][all_angles.index(tmp2)] = -1
            matrix_eq.append(0)
            
            tmp1 = angle_sort(angle[1][2] + angle[0][1] + angle[0][2])
            tmp2 = angle_sort(angle[1][0] + angle[1][1] + angle[0][0])
            output.append(tmp1 + " = " + tmp2)
            matrix.append([0]*len(all_angles))
            matrix[-1][all_angles.index(tmp1)] = 1
            matrix[-1][all_angles.index(tmp2)] = -1
            matrix_eq.append(0)
            #buffer_eq_list.append(convert_angle_eq(tmp1, tmp2))

    all_triangle(all_angles)
    matrix_print(print_it)
    line_matrix_print(print_it)
    eq_list = list(set(eq_list))
    if print_it:
        for item in eq_list:
            print_text(string_equation(item))
def all_cycle(graph):
    global lines, matrix, matrix_eq, points, point_pairs, all_angles
    cycles = []
    
    def findNewCycles(path):
        start_node = path[0]
        next_node = None
        
        for edge in graph:
            node1, node2 = edge
            
            if start_node in edge:
                next_node = node2 if node1 == start_node else node1
                
                if not visited(next_node, path):
                    # Continue searching for cycles by extending the current path
                    sub_path = [next_node] + path
                    findNewCycles(sub_path)
                elif len(path) > 2 and next_node == path[-1]:
                    # A cycle is found if the path closes (next_node == path[-1])
                    p = rotate_to_smallest(path)
                    inv = invert(p)
                    
                    if isNew(p) and isNew(inv):
                        cycles.append(p)
    
    def invert(path):
        # Reverse the path and rotate to get the smallest lexicographical order
        return rotate_to_smallest(path[::-1])
    
    def rotate_to_smallest(path):
        # Rotate the path to start with the smallest node for consistency
        n = path.index(min(path))
        return path[n:] + path[:n]
    
    def isNew(path):
        # Check if the path (cycle) is not already in the list of cycles
        return path not in cycles
    
    def visited(node, path):
        # Check if a node has already been visited in the current path
        return node in path
    
    for edge in graph:
        for node in edge:
            findNewCycles([node])
    
    # Optional: Sort cycles or process them further if needed
    return cycles

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using the ray-casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    # Iterate through each edge of the polygon
    px, py = polygon[0]
    for i in range(1, n + 1):
        sx, sy = polygon[i % n]
        # Check if the point is inside the y-bounds of the edge
        if min(py, sy) < y <= max(py, sy) and x <= max(px, sx):
            if py != sy:
                # Compute the x coordinate of the intersection point
                xints = (y - py) * (sx - px) / (sy - py) + px
            # If the point's x is less than the intersection x, toggle the inside flag
            if px == sx or x <= xints:
                inside = not inside
        # Move to the next edge
        px, py = sx, sy

    return inside

def perpendicular_line_intersection(segment_start, segment_end, point, tolerance=1e-9, precision=10):
    # Unpacking coordinates
    x1, y1 = segment_start
    x2, y2 = segment_end
    px, py = point

    # Step 1: Handle the case of a horizontal line (y1 == y2)
    if abs(y2 - y1) < tolerance:  # Horizontal line case
        intersection_x = px  # The perpendicular line will be vertical
        intersection_y = y1  # Same y as the horizontal line
        return round(intersection_x, precision), round(intersection_y, precision)

    # Step 2: Handle the case of a vertical line (x1 == x2)
    if abs(x2 - x1) < tolerance:  # Vertical line case
        intersection_x = x1  # The perpendicular line will be horizontal
        intersection_y = py
        return round(intersection_x, precision), round(intersection_y, precision)

    # Regular slope of the given line
    slope_segment = (y2 - y1) / (x2 - x1)
    intercept_segment = y1 - slope_segment * x1

    # Step 3: Find the slope of the perpendicular line (negative reciprocal)
    perpendicular_slope = -1 / slope_segment

    # Step 4: Find the equation of the perpendicular line passing through the point (px, py)
    intercept_perpendicular = py - perpendicular_slope * px

    # Step 5: Solve the system of equations (where the two lines intersect):
    # Line 1 (segment): y = slope_segment * x + intercept_segment
    # Line 2 (perpendicular): y = perpendicular_slope * x + intercept_perpendicular

    # Set the equations equal to each other to find the x-coordinate of the intersection:
    # slope_segment * x + intercept_segment = perpendicular_slope * x + intercept_perpendicular
    intersection_x = (intercept_perpendicular - intercept_segment) / (slope_segment - perpendicular_slope)

    # Step 6: Use the x-coordinate to find the y-coordinate
    intersection_y = slope_segment * intersection_x + intercept_segment

    # Rounding to avoid floating-point inaccuracies
    intersection_x = round(intersection_x, precision)
    intersection_y = round(intersection_y, precision)

    return (intersection_x, intersection_y)

def triangle_centroid(p1, p2, p3):
    """Calculate the centroid of a triangle given its three vertices."""
    return ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)


def is_reflex_by_circle(polygon, radius=1.0):
    """Identify reflex vertices by using circles and centroid method."""
    reflex_vertices = []
    n = len(polygon)

    for i in range(n):
        current_vertex = polygon[i]
        prev_vertex = polygon[i - 1]
        next_vertex = polygon[(i + 1) % n]
        
        # Form lines with adjacent vertices
        lines = [
            (prev_vertex, current_vertex),
            (current_vertex, next_vertex)
        ]

        # Find the intersections
        intersections = find_intersections(lines, current_vertex, radius)
        
        if len(intersections) < 2:
            continue  # Not enough intersections to form a triangle
        
        # Take the first intersection points found
        intersection1, _ = intersections[0]
        intersection2, _ = intersections[1]

        # Form the triangle
        triangle = [current_vertex, intersection1, intersection2]

        # Calculate the centroid of the triangle
        centroid = triangle_centroid(*triangle)

        # Check if the centroid is inside the polygon
        if not is_point_in_polygon(centroid, polygon):
            reflex_vertices.append(i)

    return reflex_vertices

all_tri = []

def all_triangle(all_angles):
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
    global buffer_eq_list
    global all_tri
    #all_tri = []
    cycle = all_cycle(point_pairs)
    print(point_pairs)
    new_cycle = []
    for item in cycle:
        remove_item = []
        for i in range(-2, len(item)-2, 1):
            if straight_line([item[i], item[i+1], item[i+2]]):
                remove_item.append(item[i+1])
        new_item = item
        for i in range(len(new_item)-1, -1, -1):
            if new_item[i] in  remove_item:
                new_item.pop(i)
        new_cycle.append(new_item)
    for x in new_cycle:
        #print("below")
        #print(x)
        to_remove = []
        for i in range(-2, len(x)-2, 1):
            angle = [x[i], x[i+1], x[i+2]]
            if straight_line(angle):
                to_remove.append(i)
        to_remove = sorted(to_remove)[::-1]
        #print(to_remove)
        for item in to_remove:
            x.pop(item)
        #print(x)
        #print("above")
        convex_angle = is_reflex_by_circle([points[y] for y in x])
        #if not is_convex
        #print(convex_angle)
        out = []
        v = None
        for i in range(-2, len(x)-2, 1):
            angle = [x[i], x[i+1], x[i+2]]
            tmp = [[z for z in x][y] for y in convex_angle]
            #print(tmp)
            #print(points[angle[1]])
            v = "".join([n2a(y) for y in angle])
            if angle[1] in tmp:
                out.append("(360-" + print_angle_3("".join([n2a(y) for y in angle])) + ")")
            else:
                out.append(print_angle_3("".join([n2a(y) for y in angle])))
        
        #print("JH")
        #print(" + ".join(out) + " = " + str())
        #print(out)
        if len(x) == 3:
            all_tri.append(v)
        if out == []:
            continue
        
        copy_out = copy.deepcopy(out)
        # buffer_eq_list.append(convert_angle_straight(out, 180*(len(x)-2)))
        
        out = copy.deepcopy(copy_out)
        for i in range(len(out)):
            out[i] = out[i].replace("(360-","").replace(")","")
        
        matrix.append([0]*len(all_angles))
        subtract = 0
        
        for i in range(len(out)):
            if "(360-" in copy_out[i]:
                #continue
                subtract += 360
                matrix[-1][all_angles.index(out[i])] = -1
            else:
                matrix[-1][all_angles.index(out[i])] = 1
            #matrix[-1][all_angles.index(out[1])] = 1
            #matrix[-1][all_angles.index(out[2])] = 1
        matrix_eq.append(180*(len(x)-2)-subtract)

    all_tri = list(set(all_tri))
#process()
def fix_angle_line(eq):
    global all_angles
    
    eq = tree_form(eq)
    content = list(set("".join(all_angles)))
    for item2 in itertools.permutations(content, 3):
        item2 = "".join(item2)
        if item2 not in string_equation(str_form(eq)):
            continue
        
        eq = replace(eq, convert_angle_2(item2), convert_angle_2(print_angle_3(item2)))
        eq = replace(eq, tree_form(str_form(convert_angle_2(item2)).replace("f_angle", "f_xangle")),\
                     tree_form(str_form(convert_angle_2(print_angle_3(item2))).replace("f_angle", "f_xangle")))
    for item in point_pairs:
        item = n2a(item[0])+n2a(item[1])
        eq = replace(eq, line_fx(item), line_fx(line_sort(item)))
        eq = replace(eq, line_fx(item[1]+item[0]), line_fx(line_sort(item)))
        eq = replace(eq, tree_form(str_form(line_fx(item)).replace("f_line", "f_xline")), tree_form(str_form(line_fx(line_sort(item))).replace("f_line", "f_xline")))
        eq = replace(eq, tree_form(str_form(line_fx(item[1]+item[0])).replace("f_line", "f_xline")), tree_form(str_form(line_fx(line_sort(item))).replace("f_line", "f_xline")))
    return str_form(eq)
def extend(line, point_start, distance):
    global points
    b = None
    a = points[a2n(point_start)]
    if line[0] == point_start:
        b = points[a2n(line[1])]
    else:
        b = points[a2n(line[0])]
    ba = [a[0] - b[0], a[1] - b[1]]
    length_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    unit_vector_ba = [ba[0] / length_ba, ba[1] / length_ba]
    bc = [unit_vector_ba[0] * distance, unit_vector_ba[1] * distance]
    c = [round(a[0] + bc[0]), round(a[1] + bc[1])]
    points.append(tuple(c))
    print_text("new point added")

def divide_line(line, new_val=None):
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
    a = a2n(line[0])
    b = a2n(line[1])
    if (a,b) not in point_pairs:
        a,b = b,a
        if (a,b) not in point_pairs:
            return None
    new_point = None
    if new_val is None:
        new_point = (round((points[a][0] + points[b][0])/2), round((points[a][1] + points[b][1])/2))
    else:
        new_point = new_val
    #print_text("\n")
    point_pairs.pop(point_pairs.index((a,b)))
    point_pairs.append((len(points),a))
    point_pairs.append((len(points),b))
    points.append(new_point)
    #process()

def is_point_on_line(line, point, tolerance=500):
    """Check if a point lies on a line segment within a given tolerance."""
    a = points[line[0]]
    b = points[line[1]]
    c = point
    
    # Calculate vectors AB and AC
    ab = [b[0] - a[0], b[1] - a[1]]
    ac = [c[0] - a[0], c[1] - a[1]]
    
    # Calculate the cross product to check for collinearity
    cross_product = ab[0] * ac[1] - ab[1] * ac[0]
    
    # Check if the cross product is within the tolerance (i.e., points are collinear)
    if abs(cross_product) > tolerance:
        return False
    
    # Check if the point C is within the bounds of the line segment AB
    # Use dot product to ensure the point lies between A and B
    dot_product = ab[0] * ac[0] + ab[1] * ac[1]
    
    if dot_product < 0:
        return False
    
    squared_length_ab = ab[0]**2 + ab[1]**2
    
    if dot_product > squared_length_ab:
        return False
    
    return True

def find_line_for_point(point):
    """Find all line segments that the given point lies on."""
    global point_pairs
    output = []
    for i, line in enumerate(point_pairs):
        if is_point_on_line(line, point):
            output.append(i)
    return output

def connect_point(point_ab):
    global lines
    global points
    global point_pairs
    global eq_list
    output = []
    point_a, point_b = point_ab
    point_pairs.append((a2n(point_a), a2n(point_b)))
    #print(
    inter = find_intersections_2(points, point_pairs)
    #print(inter)
    for p in inter:
        p = (round(p[0]), round(p[1]))
        item_list = find_line_for_point(p)
        #print(item_list)
        #print(p, [points[a2n(point_a)], points[a2n(point_b)]])
        #print(point_pairs[item_list[0]])
        points.append(p)
        to_remove = []
        to_add = []
        for item in item_list:
            a, b = point_pairs[item]
            to_remove.append(point_pairs.index((a,b)))
            to_add.append((len(points)-1,a))
            to_add.append((len(points)-1,b))
        a1, a2, a3, a4 = to_add[0][1], to_add[1][1], to_add[2][1], to_add[3][1]
        s1 = print_angle_4(a1, len(points)-1, a4), print_angle_4(a3, len(points)-1, a2)
        s2 = print_angle_4(a1, len(points)-1, a3), print_angle_4(a4, len(points)-1, a2)

        
        

        output.append(s1)

        output.append(s2)
        
        to_remove = sorted(to_remove)[::-1]
        for item in to_remove:
            point_pairs.pop(item)
        for item in to_add:
            point_pairs.append(item)
    
    return output
    #print_text("\n")
    #print(points)
    #print(point_pairs)
    #process()
    
def draw_triangle():
    global points
    global point_pairs
    points = [(200,900), (800,900), (500,500)]
    point_pairs = [(0,1), (1,2), (2,0)]
    #process()
def perpendicular(point, line, ver=1):
    global points
    global point_pairs
    global eq_list
    global all_angles
    output= None
    
    if ver == 1:
        output = perpendicular_line_intersection(points[a2n(line[0])], points[a2n(line[1])], points[a2n(point)])
    else:
        output = perpendicular_line_intersection2(points[a2n(line[0])], points[a2n(line[1])], points[a2n(point)])
    divide_line(line, output)
    num = len(points)-1
    
    tmp = connect_point(n2a(len(points)-1)+point)
    process(False)
    for item in tmp:
        row = [0]*len(all_angles)
        row[all_angles.index(item[0])] = 1
        row[all_angles.index(item[1])] = -1
        matrix.append(row)
        matrix_eq.append(0)
        
    eq1 = print_angle_3(point+n2a(num)+line[0])
    eq2 = print_angle_3(point+n2a(num)+line[1])
    row = [0]*len(all_angles)
    row[all_angles.index(eq1)] = 1
    matrix_eq.append(90)
    matrix.append(row)
    row = [0]*len(all_angles)
    row[all_angles.index(eq2)] = 1
    matrix_eq.append(90)
    matrix.append(row)
    
import copy

def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250))  # Resize image to fit the display area
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
def convert_form(eq):
    def num_not_var(eq):
        for index, item in enumerate(all_angles):
            eq = replace(eq, convert_angle(item), tree_form("v_"+str(index)))
        return eq
    if all(x not in eq for x in ["f_triangle", "f_xtriangle", "f_xangle", "f_congruent", "f_xcongruent", "f_line", "f_xline"]):
        return str_form(num_not_var(tree_form(eq)))
    return None
def convert_form_2(eq):
    global line_counter
    def num_not_var(eq):
        for index, item in enumerate(line_counter):
            eq = replace(eq, line_fx(index_line_matrix(index)), tree_form("v_"+str(index)))
        return eq
    if all(x not in eq for x in ["f_triangle", "f_xtriangle", "f_xangle", "f_angle", "f_congruent", "f_xcongruent", "f_xline"]):
        return str_form(num_not_var(tree_form(eq)))
    return None
def do_cpct():
    global eq_list
    global matrix
    global matrix_eq
    cpct_output = []
    for item in eq_list:
        if "congruent" in item:
            cpct_output += math_2.search(item, 3, [None, "formula-list-9/c.txt", None])
    
    cpct_output = list(set([fix_angle_line(x) for x in cpct_output]))

    for item in cpct_output:
        if "angle" in item:
            item = tree_form(item)
            s1 = print_angle_3(item.children[0].children[0].name[2:]+item.children[0].children[1].name[2:]+item.children[0].children[2].name[2:])
            s2 = print_angle_3(item.children[1].children[0].name[2:]+item.children[1].children[1].name[2:]+item.children[1].children[2].name[2:])
            if s1 == s2:
                continue
            row = [0]*len(all_angles)
            row[all_angles.index(s1)] = 1
            row[all_angles.index(s2)] = -1
            matrix.append(row)
            matrix_eq.append(0)
            #print("JH")
        else:
            item = tree_form(item)
            s1 = line_sort(item.children[0].children[0].name[2:]+item.children[0].children[1].name[2:])
            s2 = line_sort(item.children[1].children[0].name[2:]+item.children[1].children[1].name[2:])
            if s1 == s2:
                continue
            row = [0]*(len(line_counter)+2)
            a = index_line(*s1)
            b = index_line(*s2)
            row[a] = 1
            row[b] = -1
            line_matrix.append(row)
            fix_line_matrix()
    #eq_list += cpct_output
    #eq_list = list(set(eq_list))
def run_parallel_function():
    global points
    global point_pairs
    global eq_list
    global print_on
    global matrix
    global matrix_eq
    global all_tri
    #prove congruent(triangle(BCD),triangle(ECA))
    command=\
"""draw triangle
extend AC from C for 100
extend BC from C for 100
join CD
join CE
join DE
equation angle(ADE)=angle(CAB)
calculate"""
    

    command =\
"""draw triangle
equation angle(BAC)=angle(ABC)
equation angle(BCA)=angle(ABC)
calculate"""
    command =\
"""draw triangle
extend AC from C for 200
extend BC from C for 200
join CD
join CE
join DE
equation line(AC)=line(CD)
equation line(CE)=line(BC)
prove congruent(triangle(ACB),triangle(DCE))"""
    

    command =\
"""draw triangle
extend AC from C for 220
extend BC from C for 220
join AE CE CD BD
equation line(CE)=line(BC)
equation line(AC)=line(CD)
equation angle(ABD)=90
auto"""




    command =\
"""draw triangle
equation angle(BAC)=angle(ABC)
equation angle(BCA)=angle(ABC)
auto"""
    command =\
"""draw triangle
split AB
join CD
equation line(AC)=line(BC)
equation angle(ACD)=angle(BCD)
auto"""
    command =\
"""hide
draw triangle
split BC
split AC
join BE AD
equation line(CE)=line(CD)
equation line(AE)=line(CE)
equation line(CD)=line(BD)
show
auto"""
    command = command.split("\n")
    """
    """
    string = None
    while True:
        if string not in {"show", "hide"}:
            process()
        if command != []:
            string = command.pop(0)
            print_text(">>> ", "green", True, False)
            print_text(string, "blue", True, True)
        else:
            print_text("\nend of program", "green", True, True)
            return
        if string == "hide":
            print_on = False
        if string == "show":
            print_on = True
        if string[:13] == "draw triangle":
            draw_triangle()
        elif string == "auto":
            try_line_matrix()
            def two(angle1, angle2):
                list1 = list(itertools.permutations(list(angle1)))
                list2 = list(itertools.permutations(list(angle2)))
                for i in range(len(list1)):
                    for j in range(len(list2)):
                        out = proof_fx_3("".join(list1[i]), "".join(list2[j]))
                        if out is not None:
                            
                            return
            if len(all_tri) > 1:
                print(all_tri)
                for item in itertools.combinations(all_tri, 2):
                    #print(item[0], item[1])
                    two(item[0], item[1])
            output = try_matrix()
        elif string == "draw right triangle":
            points = [(100,400), (300,400), (300,200)]
            point_pairs = [(0,1), (1,2), (2,0)]
        elif string == "draw quadrilateral":
            points = [(272,47),(8,211),(289,380),(422,62)]
            point_pairs = [(0,1),(1,2),(2,3),(3,0)]
        elif string.split(" ")[0] == "perpendicular" and string.split(" ")[2] == "to":
            perpendicular(string.split(" ")[1], string.split(" ")[3])
        elif string == "calculate":
            output = try_matrix()
            eq_list = list(set(eq_list))
        elif string.split(" ")[0] == "prove":
            proof = parser_4.take_input(string.split(" ")[1])
            output = proof_fx(proof)
            if output:
                eq_list.append(output)
        elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "from" and string.split(" ")[4] == "for":
            extend(string.split(" ")[1], string.split(" ")[3], int(string.split(" ")[5]))
        elif string.split(" ")[0] == "split":
            divide_line(string.split(" ")[-1])
        elif string == "cpct":
            do_cpct()
        elif string.split(" ")[0] == "join":
            list_join = string.split(" ")[1:]
            tmp = None
            for join_iter in list_join:
                tmp = connect_point(join_iter)
            
            process(False)
            print(np.array(matrix))
            for item in tmp:
                row = [0]*len(all_angles)
                row[all_angles.index(item[0])] = 1
                row[all_angles.index(item[1])] = -1
                matrix.append(row)
                matrix_eq.append(0)
        elif string.split(" ")[0] == "equation":
            eq = fix_angle_line(parser_4.take_input(string.split(" ")[1]))
            
            if "angle" in eq:
                eq = convert_form(eq)
                if eq:
                    m, meq = find_all(eq, len(all_angles))
                    matrix.append(m)
                    matrix_eq.append(-meq)
            elif "parallel" in eq:
                eq_list.append(eq)
                proof_fx_2(eq)
            elif "line" in eq:
                eq = convert_form_2(eq)
                if eq:
                    m, meq = find_all(eq, len(line_counter))
                    line_matrix.append(m)


# Create main window
root = tk.Tk()
root.title("geometry Ai")
root.resizable(False, False)
# Create a frame for the console (text box) and scrollbar
console_frame = tk.Frame(root)
console_frame.grid(row=0, column=0, padx=10, pady=10)

# Add a scrollbar to the text console
scrollbar = Scrollbar(console_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a text widget to simulate console output
console = Text(console_frame, width=40, height=20, yscrollcommand=scrollbar.set)
console.pack(side=tk.LEFT)

# Configure the scrollbar to work with the text widget
scrollbar.config(command=console.yview)

console.tag_configure("black", foreground="black")
console.tag_configure("blue", foreground="blue")
console.tag_configure("red", foreground="red")
console.tag_configure("green", foreground="green")

# Create a frame for the image
image_frame = tk.Frame(root)
image_frame.grid(row=0, column=1, padx=10, pady=10)

# Label to display the image
image_label = tk.Label(image_frame)
image_label.pack(pady=5)

# Run the parallel function in a thread
parallel_thread = threading.Thread(target=run_parallel_function, daemon=True)
parallel_thread.start()

# Start the GUI event loop
root.mainloop()
