import taichi as ti
import taichi.math as tm
from config import particlesNb, simdimx, simdimy

@ti.dataclass
class TreeNode:
    firstChild: ti.i32
    nextSibling: ti.i32
    mass: ti.f32
    com: tm.vec2
    nodeSideLength: ti.f32

@ti.dataclass
class AABB:
    sideLength: ti.i32
    center: tm.vec2

nodes_array = TreeNode.field(shape=(2 * particlesNb))
bounding_boxes = AABB.field(shape=(2 * particlesNb))

pixels = ti.field(dtype=float, shape=(int(simdimx), int(simdimy)))
particles_coords = ti.Vector.field(2, dtype=float, shape=particlesNb)
particle_indexes = ti.field(dtype=ti.i32, shape=particlesNb)
prefix_boundaries = ti.field(dtype=ti.i32, shape=particlesNb)
particle_masses = ti.field(dtype=ti.f32, shape=particlesNb)
morton_keys = ti.field(dtype=ti.u64, shape=particlesNb)

next_node_idx = ti.field(dtype=ti.i32, shape=())
next_node_idx[None] = 0

next_box_idx = ti.field(dtype=ti.i32, shape=())
next_box_idx[None] = 0
