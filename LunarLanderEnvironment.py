import numpy as np
import math
import random
import pygame
from pygame import Vector2
import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions

#===========================================================================================
### Global variables
#===========================================================================================
DISPLAY_WIDTH, DISPLAY_HEIGHT = int(1280 * 0.8), int(720 * 0.8)
FPS = 30
COLORS = {'SPACE_GRAY': (24, 33, 51),
          'WHITE': (255, 255, 255),
          'LIGHT_BLUE': (50, 180, 240),
          'ORANGE': (255, 175, 105),
          'PINK': (215, 66, 235)
          }

#===========================================================================================
### Helper functions
#===========================================================================================
## Convert pygame coordinate to pymunk coordinate on the screen
def to_pymunk(point): ## point is a pygame Vector2 object
    return Vec2d(float(point.x), float(DISPLAY_HEIGHT-point.y)) ## Returns pymunk Vec2d object

## Convert pymunk coordinate to pygame coordinate on the screen
def to_pygame(point): ## point is a pymunk Vec2d object
    return (int(point.x), int(DISPLAY_HEIGHT-point.y)) ## Returns tuple of ints (ideally returns pygame Vector2 object)

## Convert pygame global coordinates into pymunk local coordinates relative to a pygame global center
def toRelativePymunk(center, end):
    return Vec2d(end - center) * Vec2d(1, -1)

## Convert pymunk local coordinates relative a pymunk global center into to pygame global coordinates
def toAbsolutePygame(center, end):
    pymunkGlobal = Vector2(center + end)
    pygameGlobal = to_pygame(pymunkGlobal)
    return pygameGlobal

def limit(value, maximum):
    return value if value <= maximum else maximum

def constrain(value, minimum, maximum):
    if value < minimum:
        return minimum
    elif value > maximum:
        return maximum
    else:
        return value

def constrainVec2d(vector, minMag, maxMag): ## vector is pymunk Vec2d
    mag = vector.get_length()
    if mag < minMag:
        return vector.normalized() * minMag
    elif mag > maxMag:
        return vector.normalized() * maxMag
    else:
        return vector

def interpolate(value, leftMin, leftMax, rightMin, rightMax):
    ## Constrain value between boundaries as precautionary measure
    value = constrain(value, leftMin, leftMax)

    ## Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    ## Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

#===========================================================================================
### Classes
#===========================================================================================
class Platform():
        def __init__(self, space, a, b, radius):
            self.a = a
            self.b = b
            self.radius = radius
            self.center = (a + b) / 2

            self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
            self.body.position = self.center

            startPoint = toRelativePymunk(self.center, self.a)
            endPoint = toRelativePymunk(self.center, self.b) 
            self.shape = pymunk.Segment(self.body, startPoint, endPoint, self.radius)
            self.shape.friction = 0.4

            space.add(self.shape)

class Lander():
    def __init__(self, space, initPosition):

        # self.pos = initPosition ## Position of lander's center
        # self.vel = Vector2(0, 0)
        # self.acc = Vector2(0, 0)
        self.mass = 80

        self.width = 58
        self.height = 120

        self.body = pymunk.Body()
        self.body.position = initPosition

        self.moduleShape = [Vec2d(0, 20), Vec2d(12, 10), Vec2d(18, -10), Vec2d(0, -15), Vec2d(-18, -10), Vec2d(-12, 10)]
        self.module = pymunk.Poly(self.body, self.moduleShape)
        self.module.mass = self.mass
        self.module.friction = 0.3

        self.leftLeg = pymunk.Poly(self.body, [Vec2d(-18, -10), Vec2d(-9, -12.5), Vec2d(-21, -20)])
        self.leftLeg.mass = 8
        self.leftLeg.friction = 0.8

        self.rightLeg = pymunk.Poly(self.body, [Vec2d(18, -10), Vec2d(9, -12.5), Vec2d(21, -20)])
        self.rightLeg.mass = 8
        self.rightLeg.friction = 0.8

        space.add(self.body, self.module, self.leftLeg, self.rightLeg)

        self.thrusterForceScale = self.mass / 4
        self.leftThrusterForce = Vec2d(4.6, 0.3) * self.thrusterForceScale ## For tilted side thrust
        # self.leftThrusterForce = Vec2d(14, 0) * self.thrusterForceScale ## For horizontal side thrust
        self.rightThrusterForce = Vec2d(-4.6, 0.3) * self.thrusterForceScale ## For tilted side thrust
        # self.rightThrusterForce = Vec2d(-14, 0) * self.thrusterForceScale ## For horizontal side thrust
        self.rearThrusterForce = Vec2d(0, 24) * self.thrusterForceScale
        self.frontThrusterForce = Vec2d(0, -24) * self.thrusterForceScale

        self.leftThrusterShape = [Vec2d(-14, 10-20/3), Vec2d(-15+(10-20/3), -1), Vec2d(-17, -10+20/6), Vec2d(-15-(10-20/3), 1)]
        self.rightThrusterShape = [Vec2d(14, 10-20/3), Vec2d(15-(10-20/3), -1), Vec2d(17, -10+20/6), Vec2d(15+(10-20/3), 1)]
        self.rearThrusterShape = [Vec2d(0, 0), Vec2d(-2.5, -2.5), Vec2d(0, -10), Vec2d(2.5, -2.5)]
        self.frontThrusterShape = [Vec2d(0, 0), Vec2d(-2.5, 2.5), Vec2d(0, 10), Vec2d(2.5, 2.5)]

        self.thrusterBools = {'left': False,
                              'rear': False,
                              'right': False,
                              'front': False}

        self.velArray = []

        self.color = COLORS['LIGHT_BLUE']
        self.thrusterColor = COLORS['ORANGE']
    
    def applyDefaultThrust(self, thrusters): ## thrusters is array of strings
        for thruster in thrusters:
            if thruster == 'left':
                # self.body.apply_impulse_at_local_point(self.leftThrusterForce, self.body.center_of_gravity) ## For horizontal side thrust
                self.body.apply_impulse_at_local_point(self.leftThrusterForce, Vec2d(-15, self.body.center_of_gravity.y)) ## For tilted side thrust
                self.thrusterBools['left'] = True
            elif thruster == 'rear':
                self.body.apply_impulse_at_local_point(self.rearThrusterForce, self.body.center_of_gravity)
                self.thrusterBools['rear'] = True
            elif thruster == 'right':
                # self.body.apply_impulse_at_local_point(self.rightThrusterForce, self.body.center_of_gravity) ## For horizontal side thrust
                self.body.apply_impulse_at_local_point(self.rightThrusterForce, Vec2d(15, self.body.center_of_gravity.y)) ## For tilted side thrust
                self.thrusterBools['right'] = True
            elif thruster == 'front':
                self.body.apply_impulse_at_local_point(self.frontThrusterForce, self.body.center_of_gravity)
                self.thrusterBools['front'] = True

    def displayThrusterForces(self, screen):
        for key, value in self.thrusterBools.items():
            if value:
                if key == 'left':
                    pygame.draw.polygon(screen, self.thrusterColor, [to_pygame(self.body.position + point) for point in self.leftThrusterShape])
                elif key == 'right':
                    pygame.draw.polygon(screen, self.thrusterColor, [to_pygame(self.body.position + point) for point in self.rightThrusterShape])
                elif key == 'front':
                    pygame.draw.polygon(screen, self.thrusterColor, [to_pygame(self.body.position + point) for point in self.frontThrusterShape])
                elif key == 'rear':
                    pygame.draw.polygon(screen, self.thrusterColor, [to_pygame(self.body.position + point) for point in self.rearThrusterShape])
                self.thrusterBools[key] = False
    
    def checkLanding(self, platform):
        ### Check safe
        leftLegContactPointSet = self.leftLeg.shapes_collide(platform.shape)
        rightLegContactPointSet = self.rightLeg.shapes_collide(platform.shape)
        safeLandingVelMag = 48

        if len(leftLegContactPointSet.points) and len(rightLegContactPointSet.points):
            if self.velArray[-1].get_length() > safeLandingVelMag:
                return 'crashed'
            else:
                return 'safe'
        elif len(leftLegContactPointSet.points) or len(rightLegContactPointSet.points):
            if self.velArray[-1].get_length() > safeLandingVelMag:
                return 'crashed'
        
        moduleContactPointSet = self.module.shapes_collide(platform.shape)
        if len(moduleContactPointSet.points):
            return 'crashed'
        
        return None ## Returns abortStatus

    def display(self, screen):
        a = interpolate(self.body.angle, -80, 80, math.pi/2, -math.pi/2)
        rotatedPoints = [Vec2d(x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)) for x, y in self.moduleShape]
        globalPoints = [toAbsolutePygame(self.body.position, point) for point in rotatedPoints]
        pygame.draw.polygon(screen, self.color, globalPoints)


class LunarLanderEnvironment():
    def __init__(self, gravity=-140.0, displayWidth=DISPLAY_WIDTH, displayHeight=DISPLAY_HEIGHT, fps=FPS):
        self.screen = None
        self.displayWidth, self.displayHeight = displayWidth, displayHeight
        self.draw_options = None
        self.clock = None
        self.fps = fps

        self.gravity = gravity
        self.space = None

        self.target = None

        self.lander = None

        self.platformRadius = 10
        self.platform = None

        self.stateSpace = ['desiredAccX', 'desiredAccY', 'angle', 'angVel'] ## SETTING
        self.stateSpaceSize = len(self.stateSpace)

        self.rewardSpace = ['desiredAccReward', 'angVelReward'] ## SETTING

        self.actionSpace = [[],
                            ['left'],
                            ['rear'],
                            ['right'],
                            ['left', 'rear'],
                            ['right', 'rear'],
                            # ['left', 'right'],
                            # ['left', 'rear', 'right']
        ] ## SETTING
        self.actionSpaceSize = len(self.actionSpace)

        self.reset()

    def displayMessage(self, msg, size, color, centerPosition):
        font = pygame.font.SysFont('comicsansms', size)
        textSurface = font.render(msg, True, color)
        textRect = textSurface.get_rect()
        textRect.center = centerPosition
        self.screen.blit(textSurface, textRect)

    def applyAction(self, action):
        self.lander.applyDefaultThrust(self.actionSpace[action])

    def getStateReward(self):
        posGlobal = self.lander.body.position
        dispFromTarget = self.lander.body.position - self.target

        vel = self.lander.body.velocity
        velRewardThreshold = 50
        velReward = -vel.get_length() + velRewardThreshold ## When velX is velXRewardThreshold or less, reward is +ve
        
        angle = (self.lander.body.angle + math.pi) % (math.pi * 2) - math.pi
        angVel = self.lander.body.angular_velocity
        angVelRewardThreshold = 0.5
        angVelReward = -abs(self.lander.body.angular_velocity) + angVelRewardThreshold # If lander angular vel is 0.4 or less then reward is +e

        desiredVelMaxMag = 44
        desiredVel = constrainVec2d(self.target - self.lander.body.position, -desiredVelMaxMag, desiredVelMaxMag)
        desiredAcc = desiredVel - self.lander.body.velocity
        desiredAccRewardThreshold = 40
        desiredAccReward = -desiredAcc.get_length() + desiredAccRewardThreshold ## If lander vel differs from desired vel by desiredAccRewardThreshold or less, then reward is +ve

        state = []
        for variable in self.stateSpace:
            if variable == "desiredAccY":
                state.append(desiredAcc.y)
            elif variable == "desiredVelY":
                state.append(desiredVel.y)
            elif variable == "velY":
                state.append(velY)

            elif variable == "desiredAccX":
                state.append(desiredAcc.x)
            elif variable == "desiredVelX":
                state.append(desiredVel.x)
            elif variable == "velX":
                state.append(velX)

            elif variable == "angle":
                state.append(angle * desiredAccRewardThreshold / angVelRewardThreshold)
            elif variable == "angVel":
                state.append(angVel * desiredAccRewardThreshold / angVelRewardThreshold)
        
        reward = 0
        for variable in self.rewardSpace:
            if variable == "desiredAccReward":
                reward += desiredAccReward
            elif variable == "angVelReward":
                reward += angVelReward * desiredAccRewardThreshold / angVelRewardThreshold
        
        return state, reward
        
    def getTermination(self, step, numSteps):
        done = False
        abortStatus = self.lander.checkLanding(self.platform) if self.platform else None

        if abortStatus == 'safe':
            done = True
        elif abortStatus == 'crashed':
            done = True
        elif not 0 <= self.lander.body.position.x <= self.displayWidth or not 0 <= self.lander.body.position.y <= self.displayHeight:
            done = True
            abortStatus = 'out_of_bounds'
        elif step >= numSteps:
            done = True
            abortStatus = 'time_out'
        else:
            done = False
            abortStatus = None

        return done, abortStatus

    def step(self, action, step, maxSteps):
        running = True
        info = dict()

        self.applyAction(action)

        self.space.step(0.02)
        self.lander.velArray.append(self.lander.body.velocity)

        nextState, reward = self.getStateReward()
        done, abortStatus = self.getTermination(step, maxSteps)
        info['abortStatus'] = abortStatus

        return nextState, reward, done, info
    
    def reset(self, test=False):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, self.gravity)        

        if test: ##### SETTING #####
            initPosition = Vec2d(random.uniform(self.displayWidth*0.2, self.displayWidth*0.8), random.uniform(self.displayHeight*0.2, self.displayHeight*0.8))
            self.lander = Lander(self.space, initPosition)
            # self.lander.body.velocity = Vec2d(8, 10)
            # self.lander.body.angular_velocity = -2.9
            
            self.platform = Platform(self.space, Vec2d(0+self.platformRadius, self.platformRadius), Vec2d(self.displayWidth-self.platformRadius, self.platformRadius), self.platformRadius)
            self.target = Vec2d(self.displayWidth/2, self.platformRadius) ## 20 is y dist between lander position and lander legs)
        else:
            initPosition = Vec2d(random.uniform(self.displayWidth*0.2, self.displayWidth*0.8), random.uniform(self.displayHeight*0.2, self.displayHeight*0.8))
            self.lander = Lander(self.space, initPosition)
            # self.lander.body.velocity = Vec2d(random.uniform(-50, 50), random.uniform(-50, 50))
            # self.lander.body.angular_velocity = random.uniform(-3.14, 3.14)

            self.platform = None
        
            self.target = Vec2d(self.displayWidth/2, self.displayHeight/2 - 200) ## 20 is y dist between lander position and lander legs)

        state, reward = self.getStateReward()
        return state

    #==================== Renderer ====================
    def renderInit(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.displayWidth, self.displayHeight))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        pygame.display.set_caption("Lunar lander")
        self.clock = pygame.time.Clock()

    def render(self, report):
        for event in pygame.event.get():
            pass
        self.screen.fill(COLORS['SPACE_GRAY'])
        self.displayMessage(report, 30, COLORS['WHITE'], Vector2(self.displayWidth/2, self.displayHeight * 0.15))
        self.space.debug_draw(self.draw_options)
        self.lander.displayThrusterForces(self.screen)
        pygame.draw.circle(self.screen, COLORS['PINK'], to_pygame(self.target), 4)
        # self.lander.display(self.screen)

        pygame.display.update()
        self.clock.tick(self.fps)

    def closeRender(self):
        pygame.quit()