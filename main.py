import taichi as ti
import taichi.math as tm
import random
from taichi.algorithms import parallel_sort
import sys
print(sys.getrecursionlimit())

ti.init(arch=ti.cpu)

particlesNb = 1000
n=320
pixels = ti.field(dtype=float, shape=(n * 2, n * 2))

simdimx = float(pixels.shape[0])
simdimy = float(pixels.shape[1])

fps = 60
dt = 1.0 / fps

scale = 2147483647 # 2**31 - 1

@ti.dataclass
class TreeNode:
    firstChild: ti.i32
    nextSibling: ti.i32
    mass: ti.f32
    com: tm.vec2
    nodeSideLength: ti.f32

@ti.func
def interleave_bits(coords: tm.uvec2) -> ti.u64:
    result = ti.u64(0)
    for i in range(32):
        result |= ((ti.u64(coords.x) >> i) & 1) << (2 * i + 1)
        result |= ((ti.u64(coords.y) >> i) & 1) << (2 * i)
    return result

@ti.func
def normalizeCoords(coords: tm.vec2) -> tm.uvec2:
    x = (max(min(coords.x, simdimx), 0.0) / simdimx)
    y = (max(min(coords.y, simdimy), 0.0) / simdimy)
    x = ti.i32(x * scale)
    y = ti.i32(y * scale)
    return tm.ivec2(x, y)

particles_coords = ti.Vector.field(2, dtype=float, shape=particlesNb)
morton_keys = ti.field(dtype=ti.u64, shape=particlesNb)
particle_indexes = ti.field(dtype=ti.i32, shape=particlesNb)
prefix_boundaries = ti.field(dtype=ti.i32, shape=particlesNb)
particle_masses = ti.field(dtype=ti.f32, shape=particlesNb)
nodes_array = TreeNode.field(shape=(2 * particlesNb))
next_node_idx = ti.field(dtype=ti.i32, shape=())
next_node_idx[None] = 0

for i in range(particlesNb):
    particles_coords[i] = [random.uniform(100.0, simdimx - 100.0), random.uniform(100.0, simdimy - 100.0)]
    particle_masses[i] = random.uniform(10, 100.0)

@ti.func
def computePrefixBoundaries(depth: int, start: int, end: int):
    start = max(0, start)
    end = min(morton_keys.shape[0], end)
    for i in range(start, end):
        if (i == start or (morton_keys[i] >> (64 - depth)) != (morton_keys[i - 1] >> (64 - depth))):
            prefix_boundaries[i] = 1
        else:
            prefix_boundaries[i] = 0

@ti.kernel
def computeKeys():
    for i in particles_coords:
        normCoords = normalizeCoords(particles_coords[i])
        morton_keys[i] = interleave_bits(normCoords)
        particle_indexes[i] = i

gui = ti.GUI("Gadget2", res=(n * 2, n * 2))

@ti.func
def recursiveBuild(start, end, depth=4):
    parent_node_idx = -1
    clusterSize = end - start
    if clusterSize <= 0:
        parent_node_idx = -1
    elif (clusterSize == 1):
        com = particles_coords[particle_indexes[start]]
        mass = particle_masses[particle_indexes[start]]
        parent_node_idx = next_node_idx[None]
        ti.atomic_add(next_node_idx[None], 1)
        nodes_array[parent_node_idx] = TreeNode(firstChild=start, nextSibling=-1, mass=mass, com=com, nodeSideLength=0)
    else:
        computePrefixBoundaries(depth, start=start, end=end)
        left = start
        firstChild = -1
        previousSibling = -1
        currentCoord = particles_coords[particle_indexes[start]]
        minx = currentCoord.x
        maxx = currentCoord.x
        miny = currentCoord.y
        maxy = currentCoord.y

        total_mass = 0.0
        node_com = tm.vec2(0.0, 0.0)
        childNb = 0

        for i in range(start + 1, end):
            if prefix_boundaries[i] == 1:
                node_idx = recursiveBuild(left, i, depth=(depth+2))
                childNb += 1
                if (previousSibling > -1):
                    nodes_array[previousSibling].nextSibling = node_idx
                else:
                    firstChild = node_idx
                previousSibling = node_idx
                left = i

                # Reset Data
                total_mass = 0.0
                node_com = tm.vec2(0.0, 0.0)
                currentCoord = particles_coords[particle_indexes[i]]
                minx = currentCoord.x
                maxx = currentCoord.x
                miny = currentCoord.y
                maxy = currentCoord.y
            else:
                # Update data
                currentCoord = particles_coords[particle_indexes[i]]
                minx = ti.min(currentCoord.x, minx)
                maxx = ti.max(currentCoord.x, maxx)
                miny = ti.min(currentCoord.y, miny)
                maxy = ti.max(currentCoord.y, maxy)
                total_mass += nodes_array[node_idx].mass
                node_com += (nodes_array[node_idx].com * nodes_array[node_idx].mass)

        if left < end:
            node_idx = recursiveBuild(left, i, depth=(depth+2))
            total_mass += nodes_array[node_idx].mass
            node_com += (nodes_array[node_idx].com * nodes_array[node_idx].mass)
            childNb += 1
            if (previousSibling > -1):
                nodes_array[previousSibling].nextSibling = node_idx
            else:
                firstChild = node_idx

        node_com /= total_mass
        nodeSideLength=ti.max(maxx - minx, maxy - miny)
        parent_node_idx = next_node_idx[None]
        ti.atomic_add(next_node_idx[None], 1)
        nodes_array[parent_node_idx] = TreeNode(firstChild=firstChild, nextSibling=-1, mass=total_mass,
                                        com=node_com, nodeSideLength=nodeSideLength)
    return parent_node_idx

@ti.kernel
def constructQuadtree(startingDepth: int):
    recursiveBuild(0, particlesNb, startingDepth)

@ti.kernel
def paint(i: float):
    oldindex = int(i - 1) if i - 1 >= 0 else particlesNb - 1
    oldcoord = particles_coords[particle_indexes[oldindex]]
    coord = particles_coords[particle_indexes[int(i)]]
    pixels[int(oldcoord.x), int(oldcoord.y)] = 0
    pixels[int(coord.x), int(coord.y)] = float(particlesNb - 1 - i) / float(particlesNb)

    # for index in particle_indexes:
    #     coord = particles_coords[particle_indexes[index]]
    #     pixels[int(coord.x), int(coord.y)] = float(index) / float(particlesNb)

depth=8

computeKeys()
parallel_sort(morton_keys, particle_indexes)
# computePrefixBoundaries(0, particlesNb, depth)
constructQuadtree(depth)

print(particles_coords)
print(particle_indexes)

gui.fps_limit = fps

i = 0.0
while gui.running:
    gui.clear(0x000000)
    
    # Draw particles (green = regular, red = boundary)
    for i in range(particlesNb):
        pos = particles_coords[particle_indexes[i]] / ti.Vector([simdimx, simdimy])
        if prefix_boundaries[i] % 5 == 0:
            color = 0xFF0000
        elif prefix_boundaries[i] % 5 == 1:
            color = 0x00FF00
        elif prefix_boundaries[i] % 5 == 2:
            color = 0x0000FF
        elif prefix_boundaries[i] % 5 == 3:
            color = 0x00FFFF
        else:
            color = 0xFF00FF
        gui.circle(pos, radius=2, color=color)  # Green = same prefix
    
    # Display depth info
    gui.text(f"Depth: {depth}", (0.05, 0.95))
    
    # Adjust depth with arrow keys
    if gui.is_pressed(ti.GUI.UP):
        depth = min(depth + 1, 16)
        computePrefixBoundaries(depth)
    if gui.is_pressed(ti.GUI.DOWN):
        depth = max(depth - 1, 1)
        computePrefixBoundaries(depth)
    gui.show()
    # paint(i)
    # gui.set_image(pixels)
    # gui.show()
    # i += dt * 2
    # if (i > particlesNb - 1):
    #     i = 0