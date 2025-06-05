import taichi as ti
import taichi.math as tm
import random
from taichi.algorithms import parallel_sort
import sys
from config import particlesNb, simdimx, simdimy, treshold, G, eps, MAX_STACK_SIZE
from fields import particles_coords, particle_indexes, prefix_boundaries, particle_masses, morton_keys, nodes_array, bounding_boxes, next_box_idx, next_node_idx
from fields import TreeNode, AABB

scale = 2147483647 # 2**31 - 1

stack_phase = ti.field(ti.i32, shape=MAX_STACK_SIZE)
stack_start = ti.field(ti.i32, shape=MAX_STACK_SIZE)
stack_end = ti.field(ti.i32, shape=MAX_STACK_SIZE)
stack_depth = ti.field(ti.i32, shape=MAX_STACK_SIZE)
stack_childNb = ti.field(ti.i32, shape=MAX_STACK_SIZE)
stack_minx = ti.field(ti.f32, shape=MAX_STACK_SIZE)
stack_maxx = ti.field(ti.f32, shape=MAX_STACK_SIZE)
stack_miny = ti.field(ti.f32, shape=MAX_STACK_SIZE)
stack_maxy = ti.field(ti.f32, shape=MAX_STACK_SIZE)

result_stack = ti.field(ti.i32, shape=MAX_STACK_SIZE)
stack_ptr = ti.field(ti.i32, shape=())
result_ptr = ti.field(ti.i32, shape=())

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

@ti.kernel
def computeKeys():
    for i in particles_coords:
        normCoords = normalizeCoords(particles_coords[i])
        morton_keys[i] = interleave_bits(normCoords)
        particle_indexes[i] = i

@ti.func
def computePrefixBoundaries(depth: int, start: int, end: int):
    segmented = False
    start = max(0, start)
    end = min(morton_keys.shape[0], end)
    for i in range(start, end):
        if (i == start):
            prefix_boundaries[i] = 1
        elif (morton_keys[i] >> (64 - depth)) != (morton_keys[i - 1] >> (64 - depth)):
            prefix_boundaries[i] = 1
            segmented = True
        else:
            prefix_boundaries[i] = 0
    return segmented

@ti.func
def push_task(phase, start, end, depth, childNb, minx, maxx, miny, maxy):
    ptr = ti.atomic_add(stack_ptr[None], 1)
    stack_phase[ptr] = phase
    stack_start[ptr] = start
    stack_end[ptr] = end
    stack_depth[ptr] = depth
    stack_childNb[ptr] = childNb
    stack_minx[ptr] = minx
    stack_maxx[ptr] = maxx
    stack_miny[ptr] = miny
    stack_maxy[ptr] = maxy

@ti.func
def pop_task():
    ptr = ti.atomic_sub(stack_ptr[None], 1) - 1
    return (
        stack_phase[ptr], stack_start[ptr], stack_end[ptr], stack_depth[ptr],
        stack_childNb[ptr], stack_minx[ptr], stack_maxx[ptr], stack_miny[ptr], stack_maxy[ptr]
    )

@ti.func
def iterativeBuild(start, end, depth):
    stack_ptr[None] = 0
    result_ptr[None] = 0

    next_node_idx[None] = 0
    next_box_idx[None] = 0
    parent_idx = 0
    box_idx = 0

    push_task(0, start, end, depth, 0, 0.0, 0.0, 0.0, 0.0)

    while (stack_ptr[None] > 0):
        phase, start, end, depth, childNb, minx, maxx, miny, maxy = pop_task()

        if phase == 0:
            clusterSize = end - start
            if clusterSize <= 0:
                result_stack[result_ptr[None]] = -1
                result_ptr[None] += 1
                continue
            elif clusterSize == 1:
                # Create leaf node
                com = particles_coords[particle_indexes[start]]
                mass = particle_masses[particle_indexes[start]]
                parent_idx = ti.atomic_add(next_node_idx[None], 1)
                nodes_array[parent_idx] = TreeNode(
                    firstChild=particle_indexes[start], nextSibling=-1,
                    mass=mass, com=com, nodeSideLength=-1
                )
                result_stack[result_ptr[None]] = parent_idx
                result_ptr[None] += 1
            else:
                # Calculate cluster boundaries and children
                childNb = 0
                cluster_left = ti.Vector([0]*8)
                cluster_right = ti.Vector([0]*8)
                initialCoords = particles_coords[particle_indexes[start]]
                minx = initialCoords.x
                maxx = initialCoords.x
                miny = initialCoords.y
                maxy = initialCoords.y

                if not computePrefixBoundaries(depth, start, end):
                    mid = start + (end - start) // 2
                    cluster_left[0] = start
                    cluster_right[0] = mid
                    cluster_left[1] = mid
                    cluster_right[1] = end
                    childNb = 2
                    for i in range(start, end):
                        currentCoord = particles_coords[particle_indexes[i]]
                        minx = ti.min(minx, currentCoord.x)
                        maxx = ti.max(maxx, currentCoord.x)
                        miny = ti.min(miny, currentCoord.y)
                        maxy = ti.max(maxy, currentCoord.y)
                else:
                    left = start
                    for i in range(start, end):
                        currentCoord = particles_coords[particle_indexes[i]]
                        if i > start and prefix_boundaries[i] == 1:
                            cluster_left[childNb] = left
                            cluster_right[childNb] = i
                            childNb += 1
                            left = i
                        minx = ti.min(minx, currentCoord.x)
                        maxx = ti.max(maxx, currentCoord.x)
                        miny = ti.min(miny, currentCoord.y)
                        maxy = ti.max(maxy, currentCoord.y)
                    if left < end:
                        cluster_left[childNb] = left
                        cluster_right[childNb] = end
                        childNb += 1

                # Push aggregation phase first
                push_task(1, -1, -1, -1, childNb, minx, maxx, miny, maxy)
                
                # Push children in reverse order
                child = childNb - 1
                while child > -1:
                    c_start = cluster_left[child]
                    c_end = cluster_right[child]
                    push_task(0, c_start, c_end, depth+2, 0, 0.0, 0.0, 0.0, 0.0)
                    child -= 1

        else:
            # Aggregate phase
            total_mass = 0.0
            node_com = ti.Vector([0.0, 0.0])
            firstChild = -1
            previousSibling = -1
            
            # Pop children from result stack
            children = ti.Vector([-1]*8)
            for i in range(childNb):
                result_ptr[None] -= 1
                children[i] = result_stack[result_ptr[None]]
            
            # Process children in original order
            for i in range(childNb):
                child_idx = children[i]
                child_node = nodes_array[child_idx]
                total_mass += child_node.mass
                node_com += child_node.com * child_node.mass
                if firstChild == -1:
                    firstChild = child_idx
                else:
                    nodes_array[previousSibling].nextSibling = child_idx
                previousSibling = child_idx
            
            if previousSibling != -1:
                nodes_array[previousSibling].nextSibling = -1

            if total_mass > 0:
                node_com /= total_mass
            
            nodeSideLength = ti.max(maxx - minx, maxy - miny)
            nodeCenter = ti.Vector([(minx + maxx) / 2, (miny + maxy) / 2])
            parent_idx = ti.atomic_add(next_node_idx[None], 1)
            nodes_array[parent_idx] = TreeNode(
                firstChild=firstChild, nextSibling=-1,
                mass=total_mass, com=node_com, nodeSideLength=nodeSideLength, center=nodeCenter
            )
            
            # Create bounding box
            box_idx = ti.atomic_add(next_box_idx[None], 1)
            bounding_boxes[box_idx] = AABB(
                sideLength=nodeSideLength,
                topleft=ti.Vector([minx, miny])
            )
            
            result_stack[result_ptr[None]] = parent_idx
            result_ptr[None] += 1

    master_node_index = result_stack[0] if result_ptr[None] > 0 else -1
    return (master_node_index, next_box_idx[None])

@ti.func
def nodeContainsParticle(node_idx, coords):
    target_center = nodes_array[node_idx].center
    sideLength = nodes_array[node_idx].nodeSideLength
    result = (coords.x >= target_center.x - sideLength / 2 and coords.x <= target_center.x + sideLength / 2
              and coords.y >= target_center.y - sideLength / 2 and coords.y <= target_center.y + sideLength / 2)
    return result

@ti.func
def computeGravityForParticle(particleIdx, masterNodeIndex):
    stack = ti.Vector([0] * MAX_STACK_SIZE)
    stack_top = 0
    particle_pos = particles_coords[particleIdx]

    total_force = tm.vec2(0.0, 0.0)

    stack[stack_top] = masterNodeIndex
    stack_top += 1
    while (stack_top > 0):
        stack_top -= 1
        node_idx = stack[stack_top]
        target_com = nodes_array[node_idx].com
        r = target_com - particle_pos
        dist = r.norm()
        if (nodes_array[node_idx].firstChild == -1):
            continue

        if nodes_array[node_idx].nodeSideLength == -1:
            if nodes_array[node_idx].firstChild != particleIdx:
                total_force += G * nodes_array[node_idx].mass * r / (dist**2 + eps**2)**1.5
        else:
            if (nodes_array[node_idx].nodeSideLength / (dist + eps) < treshold and not nodeContainsParticle(node_idx, particle_pos)):
                total_force += G * nodes_array[node_idx].mass * r / (dist**2 + eps**2)**1.5
            else:
                child = nodes_array[node_idx].firstChild
                while (child != -1):
                    stack[stack_top] = child
                    stack_top += 1
                    child = nodes_array[child].nextSibling

    return total_force
