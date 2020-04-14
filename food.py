from turtle import Turtle
# from config import game
import random as rnd
# from ai_snake import snake


class Food(Turtle):
    def __init__(self, game, snake, wn, game_map, get_cords):
        super().__init__()
        self.shape('square')
        self.color('red')
        self.name = 'Food'
        self.penup()
        self.speed(0)
        self.game = game
        self.snake = snake
        self.wn = wn
        self.game_map = game_map
        # self.objects = objects
        self.get_cords = get_cords
        self.place_food()

    # set random food position
    def place_food(self):

        dist = self.game.max_dist

        while True:
            x = rnd.randrange(-dist, dist+1, self.game.move_step)
            y = rnd.randrange(-dist, dist+1, self.game.move_step)

            # check if collision with snake head
            if (self.snake.distance(x, y) == 0):
                continue

            # check if body exist
            blocking = False
            for bodyPart in self.snake.body:
                if (bodyPart.pos() == (x, y)):
                    # print('Food gen onto object')
                    blocking = True
                    break
            if not blocking:
                break

        # place food
        self.setpos(x, y)

        self.showturtle()
        self.wn.update()


# food = Food()
