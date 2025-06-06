import taichi as ti
import taichi.math as tm
from taichi.algorithms import parallel_sort
import sys
ti.init(arch=ti.cpu)
from config import particlesNb, simdimx, simdimy, fps, defaultdt, resx, resy, margin, minMass, maxMass, radius, centerx, centery, G, distribCoef, centralObjectMass

from fields import pixels, particles_coords, particle_indexes, particle_masses, particles_vel
from fields import morton_keys, next_box_idx
from fields import timesteps, newstepsbuf
from fields import TreeNode, AABB
from integration import applyGravity
from sph import solveEquationOfState, computePressureForces

print(sys.getrecursionlimit())

from quadtree import iterativeBuild, computeKeys

timesteps.fill(-1)
newstepsbuf.fill(-1)

@ti.func
def randgen(min, max):
    return min + ti.random(float) * (max - min)

@ti.kernel
def spawnParticles():
    if (len(sys.argv) <= 1 or sys.argv[1] == "square"):
        for i in range(particlesNb):
            particles_coords[i] = [randgen(margin, simdimx - margin), randgen(margin, simdimy - margin)]
            particle_masses[i] = randgen(minMass, maxMass)
            timesteps[3, i] = i
    elif (len(sys.argv) > 1 and sys.argv[1] == "circle"):
        midrad = radius / 2.0
        for i in range(particlesNb):
            coef = ti.tanh(ti.randn() * distribCoef)
            centerDist = ti.max(midrad + coef * midrad, 10.0)
            angle = ti.random(float) * 2 * tm.pi
            relativePos = tm.vec2(tm.cos(angle), tm.sin(angle)) * centerDist
            particles_coords[i] = tm.vec2(relativePos.x + centerx, relativePos.y + centery)
            particle_masses[i] = ti.random(float) * (maxMass - minMass) + minMass
            tangent = tm.vec2(relativePos.y, -relativePos.x).normalized()
            speed = tm.sqrt((G * particle_masses[i]) / centerDist) * 10
            particles_vel[i] = tangent * speed
            timesteps[3, i] = i

    if (len(sys.argv) > 2 and sys.argv[2] == "--centralObject"):
        mid = int(particlesNb / 2)
        particles_coords[mid] = [simdimx / 2, simdimy / 2]
        particle_masses[mid] = centralObjectMass
        particles_vel[mid] = [0, 0]

gui = ti.GUI("Gadget2", res=(resx, resy))

boxNb = 0

@ti.kernel
def constructQuadtree(startingDepth: int) -> int:
    masterNoneIndex, boundingBoxNb = iterativeBuild(0, particlesNb, startingDepth)
    return masterNoneIndex

depth=4

rest_density = 0.0

gui.fps_limit = fps

pause = False
firstIt = True

spawnParticles()

deltaTime = defaultdt
it = 0
while gui.running:
    gui.clear(0x000000)

    for e in gui.get_events(gui.PRESS, gui.RELEASE):
        if e.type == gui.RELEASE and (e.key == 'p' or e.key == ti.GUI.SPACE):
            pause = not pause
        elif e.key == ti.GUI.LEFT:
            if (e.type == gui.PRESS):
                deltaTime = 0.25 / fps
            else:
                deltaTime = defaultdt
        elif e.key == ti.GUI.RIGHT:
            if (e.type == gui.PRESS):
                deltaTime = 4 / fps
            else:
                deltaTime = defaultdt

    if (not pause):
        computeKeys()
        parallel_sort(morton_keys, particle_indexes)
        masterNodeIndex = constructQuadtree(depth)
        boxNb = next_box_idx[None]
        applyGravity(masterNodeIndex, it, deltaTime)
        if firstIt:
            rest_density = solveEquationOfState(masterNodeIndex, rest_density, False)
        else:
            solveEquationOfState(masterNodeIndex, rest_density, True)
        computePressureForces(masterNodeIndex, rest_density)
    
    # for i in range(boxNb):
    #     box = bounding_boxes[i]
    #     topleft = box.topleft / ti.Vector([simdimx, simdimy])
    #     bottomRight = topleft + ti.Vector([box.sideLength, box.sideLength]) / ti.Vector([simdimx, simdimy])
    #     gui.rect(topleft, bottomRight, color=0xFFFFFF)

    for i in range(particlesNb):
        pos = particles_coords[i] / ti.Vector([simdimx, simdimy])
        gui.circle(pos, radius=2, color=0xFFFFFF)
    # gui.set_image(pixels)
    
    gui.text(f"Depth: {depth}", (0.05, 0.95))
    gui.text(f"dt: {deltaTime}", (0.05, 0.9))
    
    gui.show()
    pixels.fill(0)

    if (it == 7):
        it = 0
    else:
        it += 1
