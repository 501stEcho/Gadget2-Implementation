particlesNb = 2000
simdimx = float(640)
simdimy = float(640)
resx = int(640)
resy = int(640)
fps = 240

# Square formation
margin = 100.0

# Galaxy formation
radius = 300.0
centerx = simdimx / 2
centery = simdimy / 2
distribCoef = 0.7

maxMass = 100.0
minMass = 10.0

defaultdt = 1.0 / fps

treshold = 0.5

eps = 0.8
G = 1.0

smoothingLength = 0.2

MAX_STACK_SIZE = 1024

pressureConstant = 1.0
