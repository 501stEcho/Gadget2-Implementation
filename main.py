import taichi as ti
import taichi.math as tm
import random
from taichi.algorithms import parallel_sort
import sys
from config import particlesNb, simdimx, simdimy, fps, dt, resx, resy

ti.init(arch=ti.cpu)

from fields import pixels, particles_coords, particle_indexes, particle_masses, morton_keys, bounding_boxes, nodes_array, next_box_idx, next_node_idx, TreeNode, AABB

print(sys.getrecursionlimit())

from quadtree import iterativeBuild, computeKeys

for i in range(particlesNb):
    particles_coords[i] = [random.uniform(100.0, simdimx - 100.0), random.uniform(100.0, simdimy - 100.0)]
    particle_masses[i] = random.uniform(10, 100.0)

gui = ti.GUI("Gadget2", res=(resx, resy))

boxNb = 0

@ti.kernel
def constructQuadtree(startingDepth: int):
    masterNoneIndex, boundingBoxNb = iterativeBuild(0, particlesNb, startingDepth)

@ti.kernel
def paint(i: float):
    oldindex = int(i - 1) if i - 1 >= 0 else particlesNb - 1
    oldcoord = particles_coords[particle_indexes[oldindex]]
    coord = particles_coords[particle_indexes[int(i)]]
    pixels[int(oldcoord.x), int(oldcoord.y)] = 0
    pixels[int(coord.x), int(coord.y)] = float(particlesNb - 1 - i) / float(particlesNb)

depth=4

computeKeys()
parallel_sort(morton_keys, particle_indexes)
constructQuadtree(depth)

print(particles_coords)
print(particle_indexes)

gui.fps_limit = fps

boxNb = next_box_idx[None]

print(f"box nb : {boxNb}")

i = 0.0
while gui.running:
    gui.clear(0x000000)
    
    for i in range(boxNb):
        box = bounding_boxes[i]
        diagonal = tm.vec2(box.sideLength / 2, box.sideLength / 2)
        topleft = (box.center - diagonal) / ti.Vector([simdimx, simdimy])
        bottomRight = (box.center + diagonal) / ti.Vector([simdimx, simdimy])
        gui.rect(topleft, bottomRight, color=0xFFFFFF)

    for i in range(particlesNb):
        pos = particles_coords[particle_indexes[i]] / ti.Vector([simdimx, simdimy])
        gui.circle(pos, radius=2, color=0xFF0000)
    
    gui.text(f"Depth: {depth}", (0.05, 0.95))
    
    gui.show()
