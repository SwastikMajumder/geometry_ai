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

from PIL import Image, ImageDraw, ImageFont

def draw_points_and_lines(points, lines, image_size=(500, 500), point_radius=5, point_color=(0, 0, 0), line_color=(255, 0, 0)):
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

def line_intersection(p1, p2, q1, q2, epsilon=0.1):
    """Find the intersection point of two line segments (p1p2 and q1q2), handling floating-point precision."""
    
    def ccw(A, B, C):
        """Check if points A, B, C are in counterclockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def is_strictly_between(A, B, C):
        """Check if point B is strictly between points A and C, considering precision."""
        return (min(A[0], C[0]) < B[0] < max(A[0], C[0])) and \
               (min(A[1], C[1]) < B[1] < max(A[1], C[1]))

    A, B, C, D = p1, p2, q1, q2
    
    # Check if lines are intersecting by using the ccw method
    if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
        denominator = (A[0] - B[0]) * (D[1] - C[1]) - (A[1] - B[1]) * (D[0] - C[0])
        
        # If denominator is close to zero, lines are parallel or collinear
        if abs(denominator) < epsilon:
            return None
        
        intersect_x = ((A[0] * B[1] - A[1] * B[0]) * (D[0] - C[0]) - (A[0] - B[0]) * (D[0] * C[1] - D[1] * C[0])) / denominator
        intersect_y = ((A[0] * B[1] - A[1] * B[0]) * (D[1] - C[1]) - (A[1] - B[1]) * (D[0] * C[1] - D[1] * C[0])) / denominator
        intersect_point = (intersect_x, intersect_y)
        
        # Ensure that the intersection point is strictly within both line segments
        if is_strictly_between(A, intersect_point, B) and is_strictly_between(C, intersect_point, D):
            return intersect_point
    
    return None


def round_point(point, precision=1e-5):
    """Round a point to a given precision."""
    return (round(point[0] / precision) * precision, round(point[1] / precision) * precision)

def find_intersections_2(points, point_pairs):
    """Find all unique intersection points of the line segments not in the points list."""
    intersections = set()
    num_points = len(points)
    
    for i in range(len(point_pairs)):
        p1 = points[point_pairs[i][0]]
        p2 = points[point_pairs[i][1]]
        for j in range(i + 1, len(point_pairs)):
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

matrix = []
matrix_eq = []
    
# Generate lines from point pairs
lines = []
import itertools
def surrounding_angle(point_name):
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
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
    global matrix
    global matrix_eq
    global points
    global point_pairs
    return calculate_triangle_area([points[x] for x in point_list]) < 10
def draw_points_and_lines(points, lines, image_size=(500, 500), point_radius=5, point_color=(0, 0, 0), line_color=(255, 0, 0), text_color=(0, 0, 0)):
    # Create a white image
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Optionally, load a font for text (adjust the path and size as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # You can replace "arial.ttf" with the path to any TTF font file
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
def print_diagram():
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
    
    # Draw points and lines on the image
    image = draw_points_and_lines(points, lines)
    
    # Save image to file
    image.save('points_and_lines_image.png')
    
    # Optionally display the image
    image.show()

  
#surrounding_angle(0)
def travel_till_end(start, step):
    global lines
    global matrix
    global matrix_eq
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
    global matrix
    global matrix_eq
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
    global matrix
    global matrix_eq
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
    global matrix
    global matrix_eq
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
    global matrix
    global matrix_eq
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
    global matrix
    global matrix_eq
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
def process():
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
    global all_angles
    global fx_call
    #print("I am prcoess")
    lines = [(points[start], points[end]) for start, end in point_pairs]
    matrix = []
    matrix_eq = []
    print_diagram() 
    output = []
    find = find_intersections_2(points, point_pairs)
    #if find != []:
    #    return None
    for i in range(len(points)):
        for angle in itertools.combinations(surrounding_angle(i), 2):
            if angle[0] != angle[1]:
                output.append(print_angle_4(angle[0], i, angle[1]))
    output = list(set(output))
    all_angles = output
    #print(all_angles)
    output = []
    for i in range(len(points)):
        for angle in all_angles:
            if straight_line([a2n(x) for x in angle]):
                #print(angle)
                output.append(angle)
    output = list(set(output))
    for x in output:
        matrix.append([0]*len(all_angles))
        matrix[-1][all_angles.index(x)] = 1
        matrix_eq.append(180)
        print(x + " = 180")
        print(line_sort(x[0]+x[1]) + " + " + line_sort(x[1]+x[2]) + " = " + line_sort(x[0]+x[2]))

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
                    print(output[-1])
    output = list(set(output))
    for angle in itertools.combinations(all_angles, 2):
        if angle[0][1] == angle[1][1] and straight_line([a2n(x) for x in angle[0]]) and straight_line([a2n(x) for x in angle[1]]):
            tmp1 = angle_sort(angle[1][0] + angle[0][1] + angle[0][2])
            tmp2 = angle_sort(angle[0][0] + angle[1][1] + angle[1][2])
            output.append(tmp1 + " = " + tmp2)
            matrix.append([0]*len(all_angles))
            matrix[-1][all_angles.index(tmp1)] = 1
            matrix[-1][all_angles.index(tmp2)] = -1
            matrix_eq.append(0)
            print(output[-1])
            
            tmp1 = angle_sort(angle[1][2] + angle[0][1] + angle[0][2])
            tmp2 = angle_sort(angle[1][0] + angle[1][1] + angle[0][0])
            output.append(tmp1 + " = " + tmp2)
            matrix.append([0]*len(all_angles))
            matrix[-1][all_angles.index(tmp1)] = 1
            matrix[-1][all_angles.index(tmp2)] = -1
            matrix_eq.append(0)
            print(output[-1])
    all_triangle(all_angles)
    for item in fx_call:
        if item == None:
            continue
        item[0](*(item[1]))
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



def all_triangle(all_angles):
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
    
    cycle = all_cycle(point_pairs)
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
        convex_angle = is_reflex_by_circle([points[y] for y in x])
        #if not is_convex
        #print(convex_angle)
        out = []
        for i in range(-2, len(x)-2, 1):
            angle = [x[i], x[i+1], x[i+2]]
            tmp = [[z for z in x][y] for y in convex_angle]
            #print(tmp)
            #print(points[angle[1]])
            if angle[1] in tmp:
                out.append("(360-" + print_angle_3("".join([n2a(y) for y in angle])) + ")")
            else:
                out.append(print_angle_3("".join([n2a(y) for y in angle])))
            
        print(" + ".join(out) + " = " + str(180*(len(x)-2)))
        for i in range(3):
            out[i] = out[i].replace("(360-","").replace(")","")
        matrix.append([0]*len(all_angles))
        matrix[-1][all_angles.index(out[0])] = 1
        matrix[-1][all_angles.index(out[1])] = 1
        matrix[-1][all_angles.index(out[2])] = 1
        matrix_eq.append(180)

def angle_eq(angle_name, val):
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
    global all_angles
    if val.isdigit():
        val = int(val)
    if len(angle_name) == 1:
        angle_name = [x for x in all_angles if x[1] == angle_name][0]
    if isinstance(val, str) and len(val) == 1:
        val = [x for x in all_angles if x[1] == val][0]
    angle_name = print_angle_3(angle_name)
    
    matrix.append([0]*len(all_angles))
    matrix[-1][all_angles.index(angle_name)] = 1
    
    if isinstance(val, str):
        if sur(val):
            val = print_angle_3(val)
        else:
            val = two_times(val)
        matrix[-1][all_angles.index(val)] = -1
        matrix_eq.append(0)
    else:
        matrix_eq.append(val)
    print(angle_name + " = " + str(val))

#process()

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
    print("new point added")

def divide_line(line):
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
    new_point = (round((points[a][0] + points[b][0])/2), round((points[a][1] + points[b][1])/2))
    print()
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
    global matrix
    global matrix_eq
    global points
    global point_pairs
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
        to_remove = sorted(to_remove)[::-1]
        for item in to_remove:
            point_pairs.pop(item)
        for item in to_add:
            point_pairs.append(item)
            """
            angles = [print_angle_4(a, len(points)-1, a2n(point_a)), print_angle_4(a, len(points)-1, a2n(point_b)),\
                      print_angle_4(b, len(points)-1, a2n(point_a)), print_angle_4(b, len(points)-1, a2n(point_b))]
            angles = list(set(angles))
            for item in itertools.combinations(angles, 2):
                print(set(list(item[0]+item[1])))
                if len(set(list(item[0]+item[1]))) == 5:
                    print(item[0] + " = " + item[1])
            """
    print()
    #print(points)
    #print(point_pairs)
    #process()
def draw_triangle():
    global points
    global point_pairs
    points = [(100,300), (400,300), (250,100)]
    point_pairs = [(0,1), (1,2), (2,0)]
    #process()
import copy
memory = []
fx_call = []
def save_state():
    global points
    global point_pairs
    global memory
    global matrix
    global matrix_eq
    memory.append(copy.deepcopy([points, point_pairs, matrix, matrix_eq]))
    
def retrive_state():
    global points
    global point_pairs
    global memory
    global matrix
    global matrix_eq
    
    if len(memory) < 1:
        print("nothing to undo")
        return
    tmp = memory.pop(-1)
    points, point_pairs, matrix, matrix_eq = tmp

def find(angle_name):
    global matrix
    global matrix_eq
    global all_angles
    orig = angle_name
    if len(angle_name) == 1:
        angle_name = [x for x in all_angles if x[1] == angle_name][0]
    angle_name = print_angle_3(angle_name)
        
    mat = np.array(matrix)

    mat_eq = np.array(matrix_eq)
    X, residuals, rank, s = np.linalg.lstsq(mat, mat_eq, rcond=None)
    for i in range(len(all_angles)):
        if all_angles[i] == angle_name:
            print(orig + " = " + str(round(X[i])))
            print()
            break

while True:
    process()
    string = input(">>> ")
    if string != "undo" and string[:4] != "find":
        save_state()
    if string[:13] == "draw triangle":
        draw_triangle()
        fx_call.append(None)
    elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "from" and string.split(" ")[4] == "for":
        extend(string.split(" ")[1], string.split(" ")[3], int(string.split(" ")[5]))
        fx_call.append(None)
    elif string.split(" ")[0] == "split":
        divide_line(string.split(" ")[-1])
        fx_call.append(None)
    elif string.split(" ")[0] == "join":
        connect_point(string.split(" ")[-1])
        fx_call.append(None)
    elif string == "undo":
        retrive_state()
        if len(fx_call) > 0:
            fx_call.pop(-1)
    elif string[:4] == "find":
        find(string.split(" ")[-1])
    elif "angle" in string and "=" in string:
        #angle_eq(string.split(" ")[1], string.split(" ")[3])
        fx_call.append([angle_eq, [string.split(" ")[1], string.split(" ")[3]]])
        #process()
        #continue
    
#divide_line("DC")
#connect_point("AG")
"""
print()
angle_eq("ADC", 90)
angle_eq("CAD", 30)
angle_eq("ACB", 90)
print()
import numpy as np
matrix = np.array(matrix)

matrx_eq = np.array(matrix_eq)
X, residuals, rank, s = np.linalg.lstsq(matrix, matrix_eq, rcond=None)
for i in range(len(all_angles)):
    print(all_angles[i] + " = " + str(X[i]))
"""
