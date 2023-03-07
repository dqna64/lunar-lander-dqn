
import pygame
from pygame import Vector2
import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions

#===========================================================================================
### Helper functions
#===========================================================================================
## Convert pygame coordinate to pymunk coordinate on the screen
def to_pymunk(point): ## point is a pygame Vector2 object
    return Vec2d(float(point.x), float(DISPLAY_HEIGHT-point.y)) ## Returns pymunk Vec2d object

## Convert pymunk coordinate to pygame coordinate on the screen
def to_pygame(point): ## point is a pymunk Vec2d object
    return (int(point.x), int(DISPLAY_HEIGHT-point.y)) ## Returns tuple of ints (ideally returns pygame Vector2 object)

## Convert coordinates of ends of a segment into points relative to pivot, from pygame coordinates to pymunk coordinates (flipped upside down)
def toRelativePymunk(center, end):
    return Vec2d(end - center) * Vec2d(1, -1)

#===========================================================================================
### Display functions
#===========================================================================================
def displayBackground(screen):
    screen.fill(COLORS['SPACE_GRAY'])

def displayMessage(screen, msg, size, color, centerPosition):
    font = pygame.font.SysFont('comicsansms', size)
    textSurface = font.render(msg, True, color)
    textRect = textSurface.get_rect()
    textRect.center = centerPosition
    screen.blit(textSurface, textRect)

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
        self.body.position = to_pymunk(self.center)

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

        self.width = 80
        self.height = 120

        self.body = pymunk.Body()
        self.body.position = initPosition

        self.module = pymunk.Poly(self.body, [Vec2d(0, 20), Vec2d(12, 10), Vec2d(18, -10), Vec2d(0, -15), Vec2d(-18, -10), Vec2d(-12, 10)])
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
        self.leftThrusterForce = Vector2(6 * self.thrusterForceScale, -20 * self.thrusterForceScale)
        self.rightThrusterForce = Vector2(-6 * self.thrusterForceScale, -20 * self.thrusterForceScale)

        self.color = COLORS['LIGHT_BLUE']
    
    def applyDefaultThrust(self, thruster):
        if thruster == 'left':
            self.body.apply_impulse_at_local_point(toRelativePymunk(Vector2(0, 0), self.leftThrusterForce), Vec2d(-15,0))
            print('left thrust')
        elif thruster == 'right':
            self.body.apply_impulse_at_local_point(toRelativePymunk(Vector2(0, 0), self.rightThrusterForce), Vec2d(15,0))
            print('right thrust')
    
    def checkLanding(self, platform):
        ### Check safe
        leftLegContactPointSet = self.leftLeg.shapes_collide(platform.shape)
        rightLegContactPointSet = self.rightLeg.shapes_collide(platform.shape)
        leftLegSafe, rightLegSafe = False, False

        if len(leftLegContactPointSet.points):
            leftLegContactDist = leftLegContactPointSet.points[0].distance
            if leftLegContactDist < 0.11:
                leftLegSafe = True
        if len(rightLegContactPointSet.points):
            rightLegContactDist = rightLegContactPointSet.points[0].distance
            if rightLegContactDist < 0.11:
                rightLegSafe = True

        if leftLegSafe and rightLegSafe:
            return False, 'safe'
        
        ### Check crash
        moduleContactPoint = self.module.shapes_collide(platform.shape)
        if len(moduleContactPoint.points):
            return False, 'crash'
        
        return True, None ## Returns running, abortStatus

    def display(self, screen):
        pygame.draw.rect(screen, self.color, (self.pos.x - self.width / 2, self.pos.y - self.height / 2, self.width, self.height))

def simulation():
    running = True
    abortStatus = None

    lander = Lander(space, Vector2(DISPLAY_WIDTH/2, DISPLAY_HEIGHT/2))
    leftThrust = False
    rightThrust = False

    platformRadius = 10
    platform = Platform(space, Vector2(0+platformRadius, DISPLAY_HEIGHT-platformRadius-10), Vector2(DISPLAY_WIDTH-platformRadius, DISPLAY_HEIGHT-platformRadius-10), platformRadius)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                quit()                

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    leftThrust = True
                elif event.key == pygame.K_RIGHT:
                    rightThrust = True
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    leftThrust = False
                elif event.key == pygame.K_RIGHT:
                    rightThrust = False

        if leftThrust:
            lander.applyDefaultThrust('left')
        if rightThrust:
            lander.applyDefaultThrust('right')
        space.step(0.02)
        running, abortStatus = lander.checkLanding(platform)

        displayBackground(screen)
        space.debug_draw(draw_options)

        pygame.display.update()
        clock.tick(FPS)
    
    space.remove(lander.body, lander.module, lander.leftLeg, lander.rightLeg)
    print(f"Abort status: {abortStatus}")
    return abortStatus

def postSimulation(status):
    running = True
    displayMessage(screen, f"Lander status: {status}", 80, COLORS['WHITE'], Vector2(DISPLAY_WIDTH/2, DISPLAY_HEIGHT *0.3))
    displayMessage(screen, "Press 'esc' to quit", 40, COLORS['WHITE'], Vector2(DISPLAY_WIDTH/2, DISPLAY_HEIGHT *0.5))
    displayMessage(screen, "Press 's' to simulate again", 40, COLORS['WHITE'], Vector2(DISPLAY_WIDTH/2, DISPLAY_HEIGHT *0.7))
    pygame.display.update()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                simulateAgain = False
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    simulateAgain = True
                    running = False
        
        clock.tick(20)
        
    return simulateAgain


#===========================================================================================
### Global variables
#===========================================================================================
DISPLAY_WIDTH, DISPLAY_HEIGHT = int(1280 * 0.8), int(720 * 0.8)
FPS = 30
COLORS = {'SPACE_GRAY': (24, 33, 51),
          'WHITE': (255, 255, 255),
          'LIGHT_BLUE': (50, 180, 240)
          }

if __name__ == '__main__':

    pygame.init()

    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption("Lunar Lander")
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, -250.0)

    simulate = True
    while simulate:
        abortStatus = simulation()
        simulate = postSimulation(abortStatus)

    pygame.quit()
    quit()