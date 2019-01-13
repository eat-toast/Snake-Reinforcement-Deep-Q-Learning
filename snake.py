# 사과 정보가 State 에 포함 되는지 확인

import random, pygame, sys
from pygame.locals import *

FPS = 8             # 초당 화면 프레임
WINDOWWIDTH = 200   # 세로 길이
WINDOWHEIGHT = 200  # 가로 길이
CELLSIZE = 40       # 뱀의 몸통 크기

assert WINDOWWIDTH % CELLSIZE == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % CELLSIZE == 0, "Window height must be a multiple of cell size."

CELLWIDTH = int(WINDOWWIDTH / CELLSIZE)  # 뱀의 가로 길이
CELLHEIGHT = int(WINDOWHEIGHT / CELLSIZE)# 뱀의 세로 길이

#             R    G    B
WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
BRIGREEN  = (150, 255, 150)
DARKGREEN = (  0, 155,   0)
DARKGRAY  = ( 40,  40,  40)
BGCOLOR = BLACK

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

HEAD = 0  # syntactic sugar: index of the worm's head
pygame.init() # pygame 초기화
FPSCLOCK = pygame.time.Clock() # 화면을 초당 몇번 출력하는지 설정
DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT)) # 세팅해둔 화면 크기만큼 window를 설정한다.
BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
pygame.display.set_caption('Snake') # 디스플레이에 Snake 라고 나오도록 한다.
#설명 출처: https://kkamikoon.tistory.com/129

episode = 0


class Snake:
    def __init__(self):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, episode
        episode = episode + 1
        self.startx = random.randint(3, 4) # 시작 지점을 랜덤하게.
        self.starty = random.randint(3, 4) # 시작 지점을 랜덤하게.
        self.wormCoords = [{'x': self.startx, 'y': self.starty},
                           {'x': self.startx - 1, 'y': self.starty},
                           {'x': self.startx - 2, 'y': self.starty}] # 시작 뱀의 길이는 3 이다. 머리로 부터 x 축으로 길게 늘어져 있는 모습
        self.direction = RIGHT # 시작 뱀의 방향은 오른쪽!
        self.totalscore = 0 # 점수는 0 점

        # Start the apple in a random place.
        self.apple = self.getRandomLocation(self.wormCoords)
        # 사과의 위치를 랜덤으로 가져온다.
        # 단, 뱀의 body 와 겹치지 않는 곳에 위치한다.

    def frameStep(self, action):
        image_data, reward, done = self.runGame(action)

        return image_data, reward, done

    def runGame(self, action):
        # action을 받아 온다.
        global episode
        pygame.event.pump()

        self.pre_direction = self.direction
        # pre_direction은 이전 방향을 가리킨다.

        # action[0] up
        # action[1] down
        # action[2] left
        # action[3] right
        if (action[0] == 1) and self.direction != DOWN:
            self.direction = UP
        elif (action[1] == 1) and self.direction != UP:
            self.direction = DOWN
        elif (action[2] == 1) and self.direction != RIGHT:
            self.direction = LEFT
        elif (action[3] == 1) and self.direction != LEFT:
            self.direction = RIGHT

        # 물리적으로 가능한 방향인지 체크!



        # 1. action 이 UP 인데 기존 방향이 DOWN 이면 불가능
        # 2. action 이 DOWN 인데 기존 방향이 UP 이면 불가능
        # 3. action 이 LEFT 인데 기존 방향이 RIGHT 이면 불가능
        # 4. action 이 RIGHT 인데 기존 방향이 LEFT 이면 불가능


        # 뱀이 자기 자신을 쳤거나, 벽을 박았는지 확인
        reward = -0.1
        done = False
        if self.wormCoords[HEAD]['x'] == -1 or self.wormCoords[HEAD]['x'] == CELLWIDTH or \
                self.wormCoords[HEAD]['y'] == -1 or self.wormCoords[HEAD]['y'] == CELLHEIGHT:
            # 뱀의 머리가 X 축 범위를 벗어 낫는지
            # 뱀의 머리가 Y 춧 범위를 멋어 낫는지

            done = True
            # self.__init__() # game over
            reward = -1
            # -1 리워드
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            return image_data, reward, done

        for self.wormBody in self.wormCoords[1:]:
            if self.wormBody['x'] == self.wormCoords[HEAD]['x'] and self.wormBody['y'] == self.wormCoords[HEAD]['y']:
                # 뱀의 머리가 몸통을 관통한다면 끝
                done = True
                # self.__init__() # game over
                reward = -1
                image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                return image_data, reward, done

        # 뱀이 사과를 먹었는지 확인
        if self.wormCoords[HEAD]['x'] == self.apple['x'] and self.wormCoords[HEAD]['y'] == self.apple['y']:
            # 뱀의 꼬리 부분을 지우지 않는다.
            self.apple = self.getRandomLocation(self.wormCoords)  # 새로운 장소에 사과를 배치한다.
            reward = 2
            self.totalscore = self.totalscore + 1
            # 총점 +1
        else:
            del self.wormCoords[-1]  # 뱀의 꼬리 부분을 지운다. --> 이동하는 효과를 보여주기 위해

        # 뱀의 몸을 하나 추가 하면서 direction 방향으로 이동한다.
        if not self.examine_direction(self.direction, self.pre_direction):
            # 물리적으로 이동 불가능한지 검사한다.
            # 불가능 하다면, 그 이전 방향으로 계속 이동한다.
            self.direction = self.pre_direction

        if self.direction == UP:
            self.newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] - 1}
        elif self.direction == DOWN:
            self.newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] + 1}
        elif self.direction == LEFT:
            self.newHead = {'x': self.wormCoords[HEAD]['x'] - 1, 'y': self.wormCoords[HEAD]['y']}
        elif self.direction == RIGHT:
            self.newHead = {'x': self.wormCoords[HEAD]['x'] + 1, 'y': self.wormCoords[HEAD]['y']}
        # UP, DOWN, LEFT, RIGHT 방향에 맞게 새로운 머리(newHead)에 뱀을 추가를 한다.

        self.wormCoords.insert(0, self.newHead) #추가
        DISPLAYSURF.fill(BGCOLOR) # 배경색은 검은색으로 설정
        # self.drawGrid()
        self.drawWorm(self.wormCoords)
        self.drawApple(self.apple)
        # self.drawScore(len(self.wormCoords) - 3)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, done

    def examine_direction(self, temp, direction):
        if direction == UP:
            if temp == DOWN:
                return False
        elif direction == RIGHT:
            if temp == LEFT:
                return False
        elif direction == LEFT:
            if temp == RIGHT:
                return False
        elif direction == DOWN:
            if temp == UP:
                return False
        return True

    def retScore(self):
        global episode
        tmp1 = self.totalscore
        tmp2 = episode
        self.__init__()
        return tmp1, tmp2

    def drawPressKeyMsg(self):
        pressKeySurf = BASICFONT.render('Press a key to play.', True, DARKGRAY)
        pressKeyRect = pressKeySurf.get_rect()
        pressKeyRect.topleft = (WINDOWWIDTH - 200, WINDOWHEIGHT - 30)
        DISPLAYSURF.blit(pressKeySurf, pressKeyRect)

    def checkForKeyPress(self):
        if len(pygame.event.get(QUIT)) > 0:
            terminate()

        keyUpEvents = pygame.event.get(KEYUP)
        if len(keyUpEvents) == 0:
            return None
        if keyUpEvents[0].key == K_ESCAPE:
            terminate()
        return keyUpEvents[0].key

    def terminate(self):
        pygame.quit()
        sys.exit()

    def getRandomLocation(self, worm):
        temp = {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}
        while self.test_not_ok(temp, worm):
            temp = {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}
        return temp

    def test_not_ok(self, temp, worm):
        for body in worm:
            if temp['x'] == body['x'] and temp['y'] == body['y']:
                return True
        return False

    def showGameOverScreen(self):
        pygame.event.get()  # clear event queue
        return

    def drawScore(self, score):
        scoreSurf = BASICFONT.render('Score: %s' % (score), True, WHITE)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (WINDOWWIDTH - 120, 10)
        DISPLAYSURF.blit(scoreSurf, scoreRect)

    def drawWorm(self, wormCoords):
        a = 0
        for coord in wormCoords:
            x = coord['x'] * CELLSIZE
            y = coord['y'] * CELLSIZE

            wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE) # Rect(left, top, width, height)
            if a == 0:
                pygame.draw.rect(DISPLAYSURF, BRIGREEN, wormSegmentRect)
            else:
                pygame.draw.rect(DISPLAYSURF, DARKGREEN, wormSegmentRect)
            a = a + 1
            wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
            pygame.draw.rect(DISPLAYSURF, GREEN, wormInnerSegmentRect)

    def drawApple(self, coord):
        x = coord['x'] * CELLSIZE
        y = coord['y'] * CELLSIZE
        appleRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(DISPLAYSURF, RED, appleRect)

    def drawGrid(self):
        for x in range(0, WINDOWWIDTH, CELLSIZE):  # draw vertical lines
            pygame.draw.line(DISPLAYSURF, DARKGRAY, (x, 0), (x, WINDOWHEIGHT))
        for y in range(0, WINDOWHEIGHT, CELLSIZE):  # draw horizontal lines
            pygame.draw.line(DISPLAYSURF, DARKGRAY, (0, y), (WINDOWWIDTH, y))