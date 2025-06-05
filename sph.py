import taichi as ti
import taichi.math as tm
from config import particlesNb, smoothingLength, MAX_STACK_SIZE, pressureConstant
from fields import particles_coords, particle_masses, particles_density, nodes_array, particles_pressure

@ti.kernel
def computePressureForces(masterNodeIndex: int, rest_density: float):
    pass

@ti.func
def computeParticlePressureForce():
    pass

@ti.kernel
def solveEquationOfState(masterNodeIndex: int, rest_density: float, computePressure: bool) -> float:
    average_density = 0.0
    for i in range(particlesNb):
        particles_density[i] = computeParticleDensity(i, masterNodeIndex)
        if computePressure == True:
            particles_pressure[i] = pressureConstant * (particles_density[i] - rest_density)
        average_density += particles_density[i]
    
    if (particlesNb  != 0):
        average_density /= particlesNb
    return average_density

@ti.func
def computeParticleDensity(particleIdx, masterNodeIndex):
    stack = ti.Vector([0] * MAX_STACK_SIZE)
    stack_top = 0
    r = particles_coords[particleIdx]

    densitySum = 0.0

    stack[stack_top] = masterNodeIndex
    stack_top += 1
    while (stack_top > 0):
        stack_top -= 1
        node_idx = stack[stack_top]

        if (nodes_array[node_idx].firstChild == -1):
            continue

        if nodes_array[node_idx].nodeSideLength == -1:
            j = nodes_array[node_idx].firstChild
            rj = particles_coords[j]
            q = (r - rj).norm() / smoothingLength
            if j != particleIdx and q < 2:
                densitySum += particle_masses[j] * cubicSplineKernel((r - rj), smoothingLength)
        else:
            child = nodes_array[node_idx].firstChild
            while (child != -1):
                rj = nodes_array[child].center
                dist = (r - rj).norm()
                q = dist / smoothingLength
                if q < 2:
                    stack[stack_top] = child
                    stack_top += 1
                child = nodes_array[child].nextSibling

    return densitySum

@ti.func
def cubicSplineKernel(r: tm.vec2, h: float) -> float:
    dist = r.norm()
    q = dist / h
    res = 15.0 / (7.0 * tm.pi * h * h)
    if q < 1.0:
        res *= 2.0/3.0 - q * q + 0.5 * q * q * q
    elif q < 2.0:
        res *= 1.0/6.0 * (2 - tm.pow(q, 3))
    else:
        res = 0.0
    return res