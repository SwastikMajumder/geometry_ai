from fractions import Fraction
alter = False
print_on= True

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
    row = [Fraction(0)]*len(line_counter)
    row[line_counter.index(line1)] = Fraction(1)
    row[line_counter.index(line2)] = Fraction(-1)
    if row in line_matrix:
        return True
    row[line_counter.index(line1)] = Fraction(-1)
    row[line_counter.index(line2)] = Fraction(1)
    if row in line_matrix:
        return True
    return False
def angle_eq(angle1, angle2):
    if angle1 == angle2:
        return True
    
    row = [Fraction(0)]*len(all_angles)
    row[all_angles.index(angle1)] = Fraction(1)
    row[all_angles.index(angle2)] = Fraction(-1)
    if row in matrix and matrix_eq[matrix.index(row)] == Fraction(0):
        return True
    row[all_angles.index(angle1)] = Fraction(-1)
    row[all_angles.index(angle2)] = Fraction(1)
    if row in matrix and matrix_eq[matrix.index(row)] == Fraction(0):
        return True
    return False

def angle_per(angle):
    row = [Fraction(0)]*len(all_angles)
    row[all_angles.index(angle)] = Fraction(1)
    if row in matrix and matrix_eq[matrix.index(row)] == Fraction(90):
        return True
    return False


def line_counter_convert():
    output = []
    for item in line_counter:
        output.append(line_sort(n2a(item[0])+n2a(item[1])))
    return output

def sss_rule(a1,a2,a3, b1,b2,b3):
    line = [line_sort(a1+a2), line_sort(b1+b2), line_sort(a2+a3), line_sort(b2+b3), line_sort(a1+a3), line_sort(b1+b3)]

    for item in line:
        if item not in line_counter_convert():
            return False
        
    return line_eq(line[0], line[1]) and line_eq(line[2], line[3]) and line_eq(line[4], line[5])
def sas_rule(a1,a2,a3, b1,b2,b3):
    line = [line_sort(a1+a2), line_sort(b1+b2), line_sort(a2+a3), line_sort(b2+b3)]
    angle = [print_angle_3(a1+a2+a3), print_angle_3(b1+b2+b3)]
       
    for item in line:
        if item not in line_counter_convert():
            return False
    for item in angle:
        if item not in all_angles:
            
            return False
    
    return line_eq(line[0], line[1]) and angle_eq(angle[0], angle[1]) and line_eq(line[2], line[3])
def aas_rule(a1,a2,a3, b1,b2,b3):
    line = [line_sort(a2+a3), line_sort(b2+b3)]
    angle = [print_angle_3(a1+a2+a3), print_angle_3(b1+b2+b3), print_angle_3(a3+a1+a2), print_angle_3(b3+b1+b2)]
    
    for item in line:
        if item not in line_counter_convert():
            return False
        
    for item in angle:
        if item not in all_angles:
            return False
    
    return line_eq(line[0], line[1]) and angle_eq(angle[0], angle[1]) and angle_eq(angle[2], angle[3])

def rhs_rule(a1, a2, a3, b1, b2, b3):
    line = [line_sort(a1+a2), line_sort(b1+b2), line_sort(a1+a3), line_sort(b1+b3)]
    angle = [print_angle_3(a1+a2+a3), print_angle_3(b1+b2+b3)]
    
    for item in line:
        if item not in line_counter_convert():
            return False
    
    for item in angle:
        if item not in all_angles:
            return False
        
    return line_eq(line[0], line[1]) and angle_eq(angle[0], angle[1]) and line_eq(line[2], line[3]) and angle_per(angle[0])
def proof_fx_3(angle1, angle2):
    global eq_list
    
    angle_1 = TreeNode("f_triangle", [tree_form("d_"+angle1[0]), tree_form("d_"+angle1[1]), tree_form("d_"+angle1[2])])
    angle_2 = TreeNode("f_triangle", [tree_form("d_"+angle2[0]), tree_form("d_"+angle2[1]), tree_form("d_"+angle2[2])])
    eq = TreeNode("f_congruent", [angle_1, angle_2])
    eq = str_form(eq)
    
    for angle in [angle1+angle2, angle2+angle1]:
        if sss_rule(*angle) or sas_rule(*angle) or aas_rule(*angle) or rhs_rule(*angle):
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

def add_angle_equality(h1, h2):
    h1 = print_angle_3(h1)
    h2 = print_angle_3(h2)
    if h1 == h2:
        return
    row = [Fraction(0)]*len(all_angles)
    row[all_angles.index(h1)] = Fraction(1)
    row[all_angles.index(h2)] = Fraction(-1)
    matrix.append(row)
    matrix_eq.append(Fraction(0))

def add_line_equality(h1, h2):
    if line_sort(h1) == line_sort(h2):
        return
    row = [Fraction(0)]*(2+len(line_counter))
    
    row[index_line(*h1)] = Fraction(1)
    row[index_line(*h2)] = Fraction(-1)
    line_matrix.append(row)
    fix_line_matrix()

def proof_fx_2(a, b):
    global eq_list
    global matrix
    global matrix_eq
    global alter
    u, v = a, b
    for item in itertools.combinations(point_pairs, 2):
        if len(set([item[0][0], item[0][1], item[1][0], item[1][1]])) == 4:
            for item2 in itertools.product(item[0], item[1]):
                if line_sort(n2a(item2[0])+n2a(item2[1])) in line_counter_convert() and line_sort(n2a(item2[0])+n2a(item2[1])) != line_sort(u) and\
                   line_sort(n2a(item2[0])+n2a(item2[1])) != line_sort(v):
                    c = None
                    d = None
                    if item[0][0] in item2:
                        c = item[0][1]
                    if item[0][1] in item2:
                        c = item[0][0]
                    if item[1][0] in item2:
                        d = item[1][1]
                    if item[1][1] in item2:
                        d = item[1][0]
                    a, b = item2
                    if (is_same_line(line_sort(n2a(a)+n2a(c)), line_sort(u)) and is_same_line(line_sort(n2a(b)+n2a(d)), line_sort(v))) or\
                       (is_same_line(line_sort(n2a(a)+n2a(c)), line_sort(v)) and is_same_line(line_sort(n2a(b)+n2a(d)), line_sort(u))):
                        tmp = find_intersection_3(points[c][0], points[c][1], points[d][0], points[d][1], points[a][0], points[a][1], points[b][0], points[b][1])
                        if tmp[1] == "intersect":
                            add_angle_equality(n2a(c)+n2a(a)+n2a(b), n2a(d)+n2a(b)+n2a(a))
                
import math
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageTk

def draw_points_and_lines(points, lines, image_size=(1500, 1500), point_radius=5, point_color=(0, 0, 0), line_color=(255, 0, 0)):
    # Create a white image
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)

    # Draw points
    for x, y in points:
        draw.ellipse((float(x - point_radius), float(y - point_radius), float(x + point_radius), float(y + point_radius)), fill=point_color)

    # Draw lines
    for (x1, y1), (x2, y2) in lines:
        draw.line([(float(x1), float(y1)), (float(x2), float(y2))], fill=line_color, width=2)

    return image


def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    try:
        if x1 == x2:  # Line 1 is vertical
            if x3 == x4:  # Line 2 is also vertical
                return None, "parallel vertical lines"
            m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None  # Check if line 2 is vertical
            x = x1  # x-coordinate of intersection is the same for both vertical lines
            y = m2 * x + (y3 - m2 * x3) if m2 is not None else None  # Get y-coordinate using line 2 equation
            return (x, y), "intersect" if y is not None else "no intersection"
        
        if x3 == x4:  # Line 2 is vertical
            m1 = (y2 - y1) / (x2 - x1)
            x = x3
            y = m1 * x + (y1 - m1 * x1)
            return (x, y), "intersect"
        
        m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None  # Slope of line 1
        m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None  # Slope of line 2

        if m1 == m2:  # Lines are parallel
            return None, "parallel lines"

        if m1 is None:  # Line 1 is vertical
            x = x1
            y = m2 * x + (y3 - m2 * x3)
        elif m2 is None:  # Line 2 is vertical
            x = x3
            y = m1 * x + (y1 - m1 * x1)
        else:
            # Normal case for non-vertical, non-parallel lines
            a = m1
            b = y1 - m1 * x1
            c = m2
            d = y3 - m2 * x3
            x = (d - b) / (a - c)
            y = a * x + b
        
        return (x, y), "intersect"
    except:
        return None, "error"


import matplotlib.pyplot as plt

# Function to plot two lines given 8 numbers (4 points)
def plot_lines(x1, y1, x2, y2, x3, y3, x4, y4):
    # Convert to float if values are Fractions
    x1, y1 = float(x1), float(y1)
    x2, y2 = float(x2), float(y2)
    x3, y3 = float(x3), float(y3)
    x4, y4 = float(x4), float(y4)
    
    # Plot the two lines
    plt.figure(figsize=(6,6))
    
    # Line 1
    plt.plot([x1, x2], [y1, y2], label="Line 1", color="blue")
    
    # Line 2
    plt.plot([x3, x4], [y3, y4], label="Line 2", color="red")
    
    # Mark the points
    plt.scatter([x1, x2], [y1, y2], color="blue")
    plt.scatter([x3, x4], [y3, y4], color="red")
    
    # Add annotations for the points
    plt.text(x1, y1, f"({x1}, {y1})", fontsize=12, ha='right')
    plt.text(x2, y2, f"({x2}, {y2})", fontsize=12, ha='left')
    plt.text(x3, y3, f"({x3}, {y3})", fontsize=12, ha='right')
    plt.text(x4, y4, f"({x4}, {y4})", fontsize=12, ha='left')
    
    # Labels and Title
    plt.title('Line Segments')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.legend()
    plt.show()


def find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4):
    # Check if any lines are vertical (undefined slope)
    if x2 == x1 and x4 == x3:
        return None, "error"  # Both lines are vertical, parallel but no intersection
    elif x2 == x1:  # First line is vertical
        x = x1
        m2 = (y4 - y3) / (x4 - x3)
        d = y3 - m2 * x3
        y = m2 * x + d
    elif x4 == x3:  # Second line is vertical
        x = x3
        m1 = (y2 - y1) / (x2 - x1)
        b = y1 - m1 * x1
        y = m1 * x + b
    else:
        # Compute slopes
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)

        # Check if lines are parallel (same slope)
        if m1 == m2:
            return None, "error"  # Parallel lines do not intersect

        # Calculate intersection point
        a = m1
        b = y1 - m1 * x1
        c = m2
        d = y3 - m2 * x3
        x = (d - b) / (a - c)
        y = a * x + b

    # Check if the intersection point is within the bounds of both line segments
    def is_within(x1, x2, x):
        return min(x1, x2) <= x <= max(x1, x2)

    if is_within(x1, x2, x) and is_within(y1, y2, y) and is_within(x3, x4, x) and is_within(y3, y4, y):
        return (x, y), "intersect"
    return None, "error"

    
def find_intersections_2(points, point_pairs):
    #print(points)
    intersections = []
    for item in itertools.combinations(point_pairs, 2):
        x1, y1 = points[item[0][0]]
        x2, y2 = points[item[0][1]]
        x3, y3 = points[item[1][0]]
        x4, y4 = points[item[1][1]]
        tmp = find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4)
        if tmp[1] == "intersect":
            intersections.append(tmp[0])
    #print(intersections)       
    filtered_intersections = [point for point in intersections if point not in points]
    #print(filtered_intersections)
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

import matplotlib.pyplot as plt

def convert_to_float(points):
    return [(float(x), float(y)) for x, y in points]

def plot_lines(x1, y1, x2, y2, x3, y3, x4, y4):
    # Convert Fraction points to float for plotting
    line1 = convert_to_float([(x1, y1), (x2, y2)])
    line2 = convert_to_float([(x3, y3), (x4, y4)])

    # Plotting the first line (extended)
    plt.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], 'b', label="Line 1 (extended)", linestyle='--')
    
    # Plotting the second line segment
    plt.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], 'r', label="Line 2 (segment)", linestyle='-')
    
    # Plotting the points
    plt.scatter([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], color='blue', zorder=5, label="Points on Line 1")
    plt.scatter([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], color='red', zorder=5, label="Points on Line 2")
    
    # Display the legend
    plt.legend()
    
    # Set equal scaling
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add labels and grid for clarity
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    # Show the plot
    plt.show()

def find_intersection_line_with_segment(x1, y1, x2, y2, x3, y3, x4, y4):
    def is_within(x1, x2, x):
        return min(x1, x2) <= x <= max(x1, x2)
    ans = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
    output = False
    if ans[1] == "intersect" and is_within(x3, x4, ans[0][0]) and is_within(y3, y4, ans[0][1]):
        output = True
    global alter
    #if alter:
        #print(output)
        #plot_lines(x1, y1, x2, y2, x3, y3, x4, y4)
    
    return output


def polygon_area(points):
    n = len(points)
    area = Fraction(0)
    for i in range(n - 1):
        area += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
    area += points[-1][0] * points[0][1] - points[-1][1] * points[0][0]
    return abs(area) / 2

def surrounding_angle(given_point):
    def is_enclosed_angle(curr, h1, h2, h3):
        return find_intersection_line_with_segment(curr[0], curr[1], h2[0], h2[1], h1[0], h1[1], h3[0], h3[1])

    # Gather the list of points connected to the given_point
    lst = []
    for line in point_pairs:
        if given_point == line[0]:
            lst.append(points[line[1]])
        elif given_point == line[1]:
            lst.append(points[line[0]])

    
    for item in itertools.permutations(lst):
        if all(is_enclosed_angle(points[given_point], item[i], item[i+1], item[i+2]) for i in range(0, len(item)-2, 1)):
            lst = list(item)
            break
    # Convert points back to their original indices
    tmp = [points.index(x) for x in lst]
    #print(given_point, tmp)
    
    return tmp


def n2a(number):
    return chr(number + ord('A'))
def straight_line_2(point_list):
    global lines
    global points
    global point_pairs
    point_list = [a2n(x) for x in point_list]
    tmp = polygon_area([points[x] for x in point_list])
    return tmp == Fraction(0)
def straight_line(point_list):
    global lines
    global points
    global point_pairs
    tmp = polygon_area([points[x] for x in point_list])
    
    return tmp == Fraction(0)
def draw_points_and_lines(points, lines, image_size=(1500, 1500), point_radius=5, point_color=(0, 0, 0), line_color=(255, 0, 0), text_color=(0, 0, 0)):
    # Create a white image
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Optionally, load a font for text (adjust the path and size as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 48)  # You can replace "arial.ttf" with the path to any TTF font file
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

def angle_sort(angle):
    if a2n(angle[0]) > a2n(angle[2]):
        angle = angle[2] + angle[1] + angle[0]
    return angle
def line_sort(line):
    if a2n(line[0]) > a2n(line[1]):
        line = line[1] + line[0]
    return line
all_angles = []

def break_equation(equation):
    sub_equation_list = [equation]
    equation = equation
    for child in equation.children: # breaking equation by accessing children
        sub_equation_list += break_equation(child) # collect broken equations
    return sub_equation_list

def row_swap(matrix, i, j):
    """ Swap row i with row j in the matrix. """
    matrix[i], matrix[j] = matrix[j], matrix[i]

def row_scale(matrix, i, scale_factor):
    """ Scale row i by scale_factor. """
    matrix[i] = [scale_factor * elem for elem in matrix[i]]

def row_addition(matrix, src_row, dest_row, scale_factor):
    """ Add scale_factor * src_row to dest_row. """
    matrix[dest_row] = [
        elem_dest + scale_factor * elem_src
        for elem_dest, elem_src in zip(matrix[dest_row], matrix[src_row])
    ]

def remove_zero_rows(matrix):
    # Filter out rows that consist entirely of zeros
    return [row for row in matrix if any(element != Fraction(0) for element in row)]

# Gauss-Jordan Elimination for matrices with Fraction entries
def gauss_jordan_elimination(matrix):
    """ Perform Gauss-Jordan elimination to transform the matrix into RREF. """
    rows = len(matrix)
    cols = len(matrix[0])
    row = 0

    for col in range(cols):  # Now we don't ignore the last column (augmented part)
        # Find the pivot row
        pivot_row = max(range(row, rows), key=lambda r: abs(matrix[r][col]))

        if matrix[pivot_row][col] == 0:
            continue  # Skip this column if the pivot is zero

        # Swap the current row with the pivot row
        if pivot_row != row:
            row_swap(matrix, row, pivot_row)

        # Scale the pivot row
        row_scale(matrix, row, Fraction(1, matrix[row][col]))

        # Eliminate the current column entries in other rows
        for r in range(rows):
            if r != row:
                row_addition(matrix, row, r, -matrix[r][col])

        row += 1
        if row >= rows:
            break

    return remove_zero_rows(matrix)
def matrix_to_list(matrix):
    return [list(item) for item in matrix]

def fix_matrix():
    global matrix
    for i in range(len(matrix)):
        if len(matrix[i]) < len(all_angles):
            matrix[i] += [Fraction(0)]*(len(all_angles)-len(matrix[i]))
            
def fix_line_matrix():
    global line_matrix
    global line_counter
    global point_pairs
    
    # Target size is len(point_pairs) + len(line_counter)
    target_size = len(line_counter)
    
    for i in range(len(line_matrix)):
        # If the row is shorter, add zeros at the end
        if len(line_matrix[i]) < target_size:
            line_matrix[i] += [Fraction(0)] * (target_size - len(line_matrix[i]))
        # If the row is longer, truncate it to the target size
        elif len(line_matrix[i]) > target_size:
            line_matrix[i] = line_matrix[i][:target_size]
            
def try_line_matrix():
    global line_matrix
    if line_matrix == []:
        return
    
    A = np.array(line_matrix, dtype=object)
    new_matrix = gauss_jordan_elimination(A)
    #print(line_counter_convert())
    for item in itertools.combinations(line_counter_convert(), 2):
        row = [Fraction(0)]*(len(line_counter))
        row[index_line(*item[0])] = Fraction(1)
        row[index_line(*item[1])] = Fraction(-1)
        old_matrix = copy.deepcopy(line_matrix)
        line_matrix.append(row)
        #
        #print_fraction_matrix_as_float(gauss_jordan_elimination(line_matrix))
        
        if matrices_equal(gauss_jordan_elimination(np.array(line_matrix, dtype=object)), new_matrix):
            
            pass
        else:
            line_matrix = copy.deepcopy(old_matrix)
    fix_line_matrix()
def print_fraction_matrix_as_float(matrix):
    """ Print a NumPy matrix of Fractions as floats. """
    # Convert Fraction matrix to float matrix
    float_matrix = np.array([[float(cell) for cell in row] for row in matrix])
    
    # Print matrix with float formatting
    np.set_printoptions(precision=6, suppress=True)  # Set precision and suppress scientific notation
    print(float_matrix)

def matrices_equal(matrix1, matrix2):
    """ Check if two matrices are exactly equal. """
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return False  # Matrices have different dimensions

    for row1, row2 in zip(matrix1, matrix2):
        if any(x != y for x, y in zip(row1, row2)):
            return False  # Found differing elements
    
    return True

def try_matrix():
    global matrix
    global matrix_eq
    global eq_list
    global all_angles
    if matrix == []:
        return

    A = np.array(matrix, dtype=object)
    B = np.array(matrix_eq, dtype=object)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    augmented_matrix = np.hstack((A, B))
    #print_fraction_matrix_as_float(augmented_matrix)
    new_matrix = gauss_jordan_elimination(augmented_matrix)
    
    for item in itertools.combinations(all_angles, 2):
        row = [Fraction(0)]*len(matrix[0])
        row[all_angles.index(item[0])] = Fraction(1)
        row[all_angles.index(item[1])] = Fraction(-1)
        matrix.append(row)
        matrix_eq.append(Fraction(0))

        #print_fraction_matrix_as_float(gauss_jordan_elimination(np.hstack((np.array(matrix, dtype=object),np.array(matrix_eq, dtype=object).reshape(-1,1)))))
        
        if matrices_equal(gauss_jordan_elimination(np.hstack((np.array(matrix, dtype=object),np.array(matrix_eq, dtype=object).reshape(-1,1)))), new_matrix):

            #print(item)
            pass
        else:
            matrix.pop(-1)
            matrix_eq.pop(-1)
    print_fraction_matrix_as_float(new_matrix)
    for item in matrix_to_list(new_matrix):
        # Count non-zero elements and their specific values
        non_zero_count = sum(1 for x in item[:-1] if x != Fraction(0))
        if non_zero_count == 1 and item.count(Fraction(1)) == 1 and len(item) == len(item[:-1]) + 1:
            # Extract the lone variable and corresponding answer
            matrix.append(item[:-1])  # Coefficients
            matrix_eq.append(item[-1])  # Augmented part
            
def line_matrix_print(print_it=True):
    def remove_duplicate_rows(matrix):
        # Convert each row to a tuple and store in a set to remove duplicates
        unique_rows = set(tuple(row) for row in matrix)
        # Convert tuples back to lists and return the result
        return [list(row) for row in unique_rows]
    
    global line_matrix
    global point_pairs
    
    fix_line_matrix()
    
    line_matrix = remove_duplicate_rows(line_matrix)
    
    string = "$"
    for i in range(len(line_matrix)):
        for j in range(len(line_matrix[i])):
            if line_matrix[i][j] != Fraction(0):
                if line_matrix[i][j] < Fraction(0):
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
        
    string = "$"
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != Fraction(0):
                if matrix[i][j] == Fraction(-1):
                    string += "-angle("+all_angles[j]+")"
                elif matrix[i][j] == Fraction(1):
                    string += "+angle("+all_angles[j]+")"
                else:
                    string += "+"+str(float(matrix[i][j]))+"*angle("+all_angles[j]+")"
        string += "=" + str(int(matrix_eq[i])) + "\n"
    string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
    if print_it and string != "":
        print_text(string, "black", False, False)
                

def process(print_it=True):
    global lines
    global points
    global point_pairs
    global all_angles
    global matrix
    global matrix_eq
    global line_matrix
    global eq_list
    global alter
    
    lines = [(points[start], points[end]) for start, end in point_pairs]
    print_diagram() 
    output = []
    find = find_intersections_2(points, point_pairs)
    
    for i in range(len(points)):
        
        for angle in itertools.combinations(surrounding_angle(i), 2):
            
            if angle[0] != angle[1]:
                
                output.append(print_angle_4(angle[0], i, angle[1]))

    
    output = list(set(output))
    append_angles = set(output) - set(all_angles)
    reject_angles = set(all_angles) - set(output)
    
    all_angles = all_angles + list(append_angles)

    index_iter = sorted([all_angles.index(item) for item in reject_angles])[::-1]
    
    #print(all_angles)
    for i in range(len(matrix)-1,-1,-1):
        for item in index_iter:
            if int(matrix[i][item]) != Fraction(0):
                matrix.pop(i)
                matrix_eq.pop(i)
                break
    for item in index_iter:
        for i in range(len(matrix)):
            matrix[i].pop(item)
        all_angles.pop(item)
    for item in point_pairs:
        s = n2a(item[0])+n2a(item[1])
        index_line(*s)
    output = []
    
    for i in range(len(points)):
        for angle in all_angles:
            if straight_line([a2n(x) for x in angle]):
                
                output.append(angle)
    output = list(set(output))
    for x in output:
        
        index_line(*line_sort(x[0]+x[1]))
        matrix.append([Fraction(0)]*len(all_angles))
        matrix[-1][all_angles.index(x)] = Fraction(1)
        matrix_eq.append(Fraction(180))
        
        row = [Fraction(0)]*(len(line_counter)+3)
        row[index_line(x[0], x[1])] = Fraction(1)
        row[index_line(x[1], x[2])] = Fraction(1)
        row[index_line(x[2], x[0])] = Fraction(-1)
        line_matrix.append(row)
        

    for angle in itertools.permutations(all_angles, 3):
        if combine(angle[0], angle[1]) == angle[2]:
            go_to_next = False
            if straight_line_2(angle[0]) or straight_line_2(angle[1]) or straight_line_2(angle[2]):
                if straight_line_2(angle[2]):
                    go_to_next = True
                else:
                    continue
            if not go_to_next:
                
                hhh = [a2n(h) for h in list(set(angle[0]+angle[1]+angle[2])) if list(angle[0]+angle[1]+angle[2]).count(h) == 3][0]
                hh = [(a2n(h), hhh) for h in list(set(angle[0]+angle[1]+angle[2])) if list(angle[0]+angle[1]+angle[2]).count(h) == 2]
                orig = copy.deepcopy(point_pairs)
                point_pairs = hh
                hh = surrounding_angle(hhh)
                point_pairs = copy.deepcopy(orig)
            
            if go_to_next or n2a(hh[1]) not in angle[2]:
                matrix.append([Fraction(0)]*len(all_angles))
                matrix[-1][all_angles.index(angle[0])] = Fraction(1)
                matrix[-1][all_angles.index(angle[1])] = Fraction(1)
                matrix[-1][all_angles.index(angle[2])] = Fraction(-1)
                matrix_eq.append(Fraction(0))
                    
    for angle in itertools.combinations(all_angles, 2):
        if angle[0][1] == angle[1][1] and straight_line([a2n(x) for x in angle[0]]) and straight_line([a2n(x) for x in angle[1]]):
            tmp1 = print_angle_3(angle[1][0] + angle[0][1] + angle[0][2])
            tmp2 = print_angle_3(angle[0][0] + angle[1][1] + angle[1][2])
            matrix.append([Fraction(0)]*len(all_angles))
            matrix[-1][all_angles.index(tmp1)] = Fraction(1)
            matrix[-1][all_angles.index(tmp2)] = Fraction(-1)
            matrix_eq.append(Fraction(0))
            
            tmp1 = print_angle_3(angle[1][2] + angle[0][1] + angle[0][2])
            tmp2 = print_angle_3(angle[1][0] + angle[1][1] + angle[0][0])
            matrix.append([Fraction(0)]*len(all_angles))
            matrix[-1][all_angles.index(tmp1)] = Fraction(1)
            matrix[-1][all_angles.index(tmp2)] = Fraction(-1)
            matrix_eq.append(Fraction(0))

    generate_all_lines()
    fix_line_matrix()
    
    all_triangle(all_angles)
    do_isoceles()
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
                    sub_path = [next_node] + path
                    findNewCycles(sub_path)
                elif len(path) > 2 and next_node == path[-1]:
                    p = rotate_to_smallest(path)
                    inv = invert(p)
                    if isNew(p) and isNew(inv):
                        cycles.append(p)
    def invert(path):
        return rotate_to_smallest(path[::-1])
    def rotate_to_smallest(path):
        n = path.index(min(path))
        return path[n:] + path[:n]
    def isNew(path):
        return path not in cycles
    def visited(node, path):
        return node in path
    for edge in graph:
        for node in edge:
            findNewCycles([node])
    return cycles



def perpendicular_line_intersection(segment_start, segment_end, point):
    x1, y1 = segment_start
    x2, y2 = segment_end
    xp, yp = point
    
    # Case 1: If the line is vertical (x1 == x2), the perpendicular line will have the same x-coordinate
    if x2 == x1:
        xq = x1
        yq = yp
    # Case 2: If the line is horizontal (y1 == y2), the perpendicular line will have the same y-coordinate
    elif y2 == y1:
        xq = xp
        yq = y1
    # Case 3: General case for a non-horizontal, non-vertical line
    else:
        # Slope of the line AB
        m = (y2 - y1) / (x2 - x1)
        # Slope of the perpendicular line
        m_perp = -1 / m
        
        # Solve for xq using the equation:
        # m(xq - x1) + y1 = m_perp(xq - xp) + yp
        xq = (m * x1 - m_perp * xp + yp - y1) / (m - m_perp)
        
        # Find yq using the equation of line AB: yq = m(xq - x1) + y1
        yq = m * (xq - x1) + y1
    
    return (xq, yq)


def is_reflex_vertex(polygon, vertex_index):
    prev_index = (vertex_index - 1) % len(polygon)
    next_index = (vertex_index + 1) % len(polygon)
    modified_polygon = polygon[:vertex_index] + polygon[vertex_index + 1:]
    original_area = polygon_area(polygon)
    modified_area = polygon_area(modified_polygon)
    if modified_area <= original_area:
        return False
    else:
        return True
def is_reflex_by_circle(polygon):
    output = []
    for i in range(len(polygon)):
        if is_reflex_vertex(polygon, i):
            output.append(i)
    return output

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
    #print(point_pairs)
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
        
        to_remove = []
        for i in range(-2, len(x)-2, 1):
            angle = [x[i], x[i+1], x[i+2]]
            if straight_line(angle):
                to_remove.append(i)
        to_remove = sorted(to_remove)[::-1]
        
        for item in to_remove:
            x.pop(item)
            
        convex_angle = is_reflex_by_circle([points[y] for y in x])
        
        out = []
        v = None
        for i in range(-2, len(x)-2, 1):
            angle = [x[i], x[i+1], x[i+2]]
            tmp = [[z for z in x][y] for y in convex_angle]
            
            v = "".join([n2a(y) for y in angle])
            if angle[1] in tmp:
                out.append("(360-" + print_angle_3("".join([n2a(y) for y in angle])) + ")")
            else:
                out.append(print_angle_3("".join([n2a(y) for y in angle])))

        
        if len(x) == 3:
            all_tri.append(v)
        if out == []:
            continue
        
        copy_out = copy.deepcopy(out)
        
        
        out = copy.deepcopy(copy_out)
        for i in range(len(out)):
            out[i] = out[i].replace("(360-","").replace(")","")
        
        matrix.append([Fraction(0)]*len(all_angles))
        subtract = 0
        
        for i in range(len(out)):
            if "(360-" in copy_out[i]:
                #continue
                subtract += 360
                matrix[-1][all_angles.index(out[i])] = Fraction(-1)
            else:
                matrix[-1][all_angles.index(out[i])] = Fraction(1)
        matrix_eq.append(Fraction(180*(len(x)-2)-subtract))

    all_tri = list(set(all_tri))

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
    c = tuple([Fraction(round(a[0] + bc[0])), Fraction(round(a[1] + bc[1]))])
    out = c
    if polygon_area([a,b,c]) != Fraction(0):
        out = perpendicular_line_intersection(a, b, c)
    points.append(out)
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
    points.append((Fraction(new_point[0]), Fraction(new_point[1])))
    #process()

def is_point_on_line(line, point, tolerance=500):
    """Check if a point lies on a line segment within a given tolerance."""
    a = points[line[0]]
    b = points[line[1]]
    c = point
    
    return polygon_area([a,b,c]) == Fraction(0)

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

    inter = find_intersections_2(points, point_pairs)
    print(inter)
    print("HHIH")
    
    for p in inter:
        
        item_list = find_line_for_point(p)
        
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

    
def draw_triangle():
    global points
    global point_pairs
    points = [(Fraction(400),Fraction(800)), (Fraction(800),Fraction(750)), (Fraction(600),Fraction(400))]

    #points = [(Fraction(-1),Fraction(0)), (Fraction(0),Fraction(1)), (Fraction(1),Fraction(0))]
    point_pairs = [(0,1), (1,2), (2,0)]
    
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
        row = [Fraction(0)]*len(all_angles)
        row[all_angles.index(item[0])] = Fraction(1)
        row[all_angles.index(item[1])] = Fraction(-1)
        matrix.append(row)
        matrix_eq.append(Fraction(0))
        
    eq1 = print_angle_3(point+n2a(num)+line[0])
    eq2 = print_angle_3(point+n2a(num)+line[1])
    row = [Fraction(0)]*len(all_angles)
    row[all_angles.index(eq1)] = Fraction(1)
    matrix_eq.append(Fraction(90))
    matrix.append(row)
    row = [Fraction(0)]*len(all_angles)
    row[all_angles.index(eq2)] = Fraction(1)
    matrix_eq.append(Fraction(90))
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
def find_all_paths(graph, start_node, end_node, path=[]):
    path = path + [start_node]
    if start_node == end_node:
        return [path]
    if start_node not in graph:
        return []
    paths = []
    for neighbor in graph[start_node]:
        if neighbor not in path:
            new_paths = find_all_paths(graph, neighbor, end_node, path)
            for p in new_paths:
                paths.append(p)
    return paths

def generate_graph():
    graph = dict()
    for i in range(len(points)):
        graph[n2a(i)] = [n2a(x) for x in surrounding_angle(i)]
    return graph

def generate_all_lines():
    for item in itertools.combinations(range(len(points)), 2):
        for path in find_all_paths(generate_graph(), item[0], item[1]):
            if straight_line_2(path):
                for item2 in itertools.combinations(path, 2):
                    index_line(item2[0], item2[1])
def is_same_line(line1, line2):
    if line1 == line2:
        return True
    for item in itertools.combinations(range(len(points)), 2):
        for path in find_all_paths(generate_graph(), item[0], item[1]):
            if straight_line_2(path) and line1[0] in path and line1[1] in path and line2[0] in path and line2[1] in path:
                return True
    return False
def do_isoceles():
    global line_matrix
    global all_angles
    global matrix
    global matrix_eq
    global all_tri
    lst = line_counter_convert()
    for item in line_matrix:
        if item.count(Fraction(0))==len(item)-2 and item.count(Fraction(1)) == 1 and item.count(Fraction(-1)) == 1:
            line1 = lst[item.index(1)]
            line2 = lst[item.index(-1)]
            for tri in all_tri:
                if line1[0] in tri and line2[0] in tri and line1[1] in tri and line2[1] in tri:
                    common = set(line1) & set(line2)
                    common = list(common)[0]
                    a = list(set(line1) - set(common))[0]
                    b = list(set(line2) - set(common))[0]
                    row = [Fraction(0)]*len(all_angles)
                    row[all_angles.index(print_angle_3(common+a+b))] = Fraction(1)
                    row[all_angles.index(print_angle_3(common+b+a))] = Fraction(-1)
                    matrix.append(row)
                    matrix_eq.append(Fraction(0))
                    break
import re
def do_cpct():
    global eq_list
    global matrix
    global matrix_eq
    cpct_output = []
    for item in eq_list:
        if "congruent" in item:
            m = re.findall(r'[A-Z]{3}', string_equation(item))
            m_list = []
            for item2 in itertools.permutations(range(3)):
                m_list.append([   m[0][item2[0]] + m[0][item2[1]] +  m[0][item2[2]],\
                                  m[1][item2[0]] + m[1][item2[1]] +  m[1][item2[2]]    ])
            for item2 in m_list:
                add_angle_equality(item2[0], item2[1])
                add_line_equality(item2[0][:-1], item2[1][:-1])
    fix_matrix()
def run_parallel_function():
    global points
    global point_pairs
    global eq_list
    global print_on
    global matrix
    global matrix_eq
    global all_tri
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
"""draw triangle
split AB
join CD
extend AB from A for 300
join AE CE
equation line(CE)=line(BC)
equation line(DE)=line(AB)
auto"""
    command =\
"""draw triangle
split AB
join CD
equation angle(ADC)=90
equation line(BD)=line(AD)
auto
auto"""
    command =\
"""draw triangle
split AB
join CD
extend CD from D for 300
join DE AE BE
equation line_eq AC BC
equation line_eq AE BE
compute"""


    command =\
"""draw triangle
extend AC from C for 400
join CD BD
split BC
join AE
extend AE to BD
join EF
compute"""
    command =\
"""hide
draw triangle
perpendicular C to AB
equation line_eq AC BC
compute
show
compute"""
    command =\
"""draw triangle
extend AC from C for 200
extend BC from C for 200
join CD DE CE
equation parallel_line AB DE
equation line_eq AC CD
compute"""
    command =\
"""hide
draw quadrilateral
join BD
equation parallel_line AB CD
equation parallel_line AD BC
compute
show
compute"""
    command =\
"""draw triangle
split BC
split AC
join AD BE
equation line_eq AE CE
equation line_eq BD CD
equation line_eq AC BC
compute
compute"""
    command =\
"""draw triangle
extend AB from A for 200
join CD AD
split AB
join CE
equation line_eq DE AB
equation line_eq CD BC
compute"""
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
        elif string == "draw quadrilateral":
            points = [(Fraction(400),Fraction(800)), (Fraction(800),Fraction(750)), (Fraction(600),Fraction(400)), (Fraction(400),Fraction(550))]
            point_pairs = [(0,1), (1,2), (2,3), (3,0)]
        elif string == "compute":
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
                #print(all_tri)
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
        elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "to":
            val = find_intersection(points[a2n(string.split(" ")[1][0])][0], points[a2n(string.split(" ")[1][0])][1],\
                                    points[a2n(string.split(" ")[1][1])][0], points[a2n(string.split(" ")[1][1])][1],\
                                    points[a2n(string.split(" ")[3][0])][0], points[a2n(string.split(" ")[3][0])][1],\
                                    points[a2n(string.split(" ")[3][1])][0], points[a2n(string.split(" ")[3][1])][1])
            print(val)
            divide_line(string.split(" ")[3], val[0])
            
        elif string.split(" ")[0] == "split":
            divide_line(string.split(" ")[-1])
        elif string == "cpct":
            do_cpct()
        elif string.split(" ")[0] == "join":
            list_join = string.split(" ")[1:]
            tmp = None
            for join_iter in list_join:
                print("hi")
                tmp = connect_point(join_iter)
            process(False)
            for item in tmp:
                row = [Fraction(0)]*len(all_angles)
                row[all_angles.index(item[0])] = Fraction(1)
                row[all_angles.index(item[1])] = Fraction(-1)
                matrix.append(row)
                matrix_eq.append(Fraction(0))
        elif string.split(" ")[0] == "equation":
            
            eq_type = string.split(" ")[1]
            if "angle_eq" == eq_type:
                a = print_angle_3(string.split(" ")[2])
                b = print_angle_3(string.split(" ")[3])
                row = [Fraction(0)]*len(all_angles)
                row[all_angles.index(a)] = Fraction(1)
                row[all_angles.index(b)] = Fraction(-1)
                matrix.append(row)
                matrix_eq.append(Fraction(0))
            elif "angle_val" == eq_type:
                a = print_angle_3(string.split(" ")[2])
                val = int(string.split(" ")[3])
                row = [Fraction(0)]*len(all_angles)
                row[all_angles.index(a)] = Fraction(1)
                matrix.append(row)
                matrix_eq.append(Fraction(val))
            elif "parallel_line" == eq_type:
                a = line_sort(string.split(" ")[2])
                b = line_sort(string.split(" ")[3])
                eq = str_form(TreeNode("f_parallel", [line_fx(line_sort(a)), line_fx(line_sort(b))]))
                eq_list.append(eq)
                proof_fx_2(a, b)
            elif "line_eq" == eq_type:
                a = line_sort(string.split(" ")[2])
                b = line_sort(string.split(" ")[3])
                row = [Fraction(0)]*(len(all_angles)+2)
                row[index_line(*a)] = Fraction(1)
                row[index_line(*b)] = Fraction(-1)
                line_matrix.append(row)
                fix_line_matrix()


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
