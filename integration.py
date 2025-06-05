
import taichi as ti
import taichi.math as tm
import random

from fields import particles_coords, particles_acc, particles_vel, binpow, timesteps, timestepsNb, timestepssize, newstepsbuf
from quadtree import computeGravityForParticle
from config import simdimx, simdimy, particlesNb

@ti.func
def clamp(val, lb, ub):
    return ti.max(lb, ti.min(val, ub))

@ti.func
def updateParticleAcceleration(index, masterNodeIndex, timestep):
    particles_coords[index] += particles_vel[index] * timestep + 0.5 * particles_acc[index] * timestep**2
    acc = computeGravityForParticle(index, masterNodeIndex)
    particles_vel[index] += 0.5 * (particles_acc[index] + acc) * timestep
    particles_acc[index] = acc

    coord = particles_coords[index]
    particles_coords[index] = tm.vec2(clamp(coord[0], 0.0, simdimx), clamp(coord[1], 0.0, simdimy))

    return acc

@ti.func
def determineFittingTimestep(acc):
    accnorm = acc.norm()

    res = 3
    if accnorm > 1e-2:
        res = 0
    elif accnorm > 1e-3:
        res = 1
    elif accnorm > 1e-4:
        res = 2

    return res

@ti.func
def assignParticleTimestep(index, it, newTimeSteps):
    acc = particles_acc[index]

    newTimeStep = determineFittingTimestep(acc)

    while (it % binpow[newTimeStep] != 0):
        newTimeStep -= 1
    
    newTimeSteps[index] = newTimeStep

@ti.kernel
def applyGravity(masterNodeIndex: int, it: int, deltaTime: float) -> int:
    for i in range(timestepsNb):
        if (it % binpow[i] == 0):
            timestep = deltaTime * binpow[i]
            for j in range(timestepssize[i]):
                updateParticleAcceleration(timesteps[i, j], masterNodeIndex, timestep)
                assignParticleTimestep(timesteps[i, j], it, newstepsbuf)
    
    timestepssize.fill(0)
    for i in range(particlesNb):
        newTimeStep = newstepsbuf[i]
        index = ti.atomic_add(timestepssize[newTimeStep], 1)
        timesteps[newTimeStep, index] = i
    return 0