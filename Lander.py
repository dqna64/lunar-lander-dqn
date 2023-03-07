class Lander():
    def __init__(self, position, color):
        self.pos = position ## Position of lander's center
        self.vel = Vector2(0, 0)
        self.acc = Vector2(0, 0)
        self.mass = 100

        self.width = 80
        self.height = 120

        self.leftThrustForce = Vector2(-40, -40)
        self.rightThrustForce = Vector2(40, -40)

        self.color = color

    def applyForce(self, force):
        self.acc += force / self.mass

    def update(self):
        self.vel += self.acc
        self.pos += self.vel

        self.acc = Vector2(0, 0)
    
    def applyDefaultThrust(self, direction):
        if direction == 'left':
            self.applyForce(self.leftThrustForce)
            print('left thrust')
        elif direction == 'right':
            self.applyForce(self.rightThrustForce)
            print('right thrust')

    def display(self, screen):
        pygame.draw.rect(screen, self.color, (self.pos.x - self.width / 2, self.pos.y - self.height / 2, self.width, self.height))
