from turtle import Turtle
from functions_of_game import *
# import random as rnd
import csv
import turtle
# from ai_snake import update_game


class Snake(Turtle):
    idCounter = 0

    # directions 0-east, 90-north, 180-west, 270-south
    def __init__(self, name, position, game, wn, game_map, objects, direction=90):
        super().__init__(shape='turtle', visible=False)
        Snake.idCounter += 1
        self.name = 'Adam{0}'.format(Snake.idCounter)
        self.color('green')
        self.shapesize(1.5, 1.5, 1)
        self.penup()
        self.setposition(position)
        self.old_xy = [0, 0]
        # self.previousDirection = 0 # 0-none, 1-Up, 2-Right, 3-Down, 4-Left
        self.setheading(direction)
        # self.setheading(180)
        self.speed(0)
        self.body = []
        # self.grow()
        self.showturtle()
        self.food_count = 0
        self.food_count_record = 0
        self.game = game
        self.wn = wn
        self.game_map = game_map
        self.objects = objects
        self.record = False  # if True after eatch move record scan and move to log.inputs data
        self.records_count = 0
        self.inputs = []
        self.model = None
        self.grow()

    def go_right(self):
        self.update_game('Right')

    def go_down(self):
        self.update_game('Down')

    def go_left(self):
        self.update_game('Left')

    def go_up(self):
        self.update_game('Up')

    # after each move update the game
    def update_game(self, direction):

        # record log to file csv if record is enabled
        if self.record:
            # make scan of objects
            inputs = self.scan(False)
            # add one input to the dictionary
            inputs['Direction'] = direction

        # change cords by direction
        if direction == 'Right':
            self.setheading(0)
            self.old_xy = self.pos()
            self._position = (self.xcor() + self.game.move_step, self.ycor())

        elif direction == 'Down':
            self.setheading(270)
            self.old_xy = self.pos()
            self._position = (self.xcor(), self.ycor() - self.game.move_step)

        elif direction == 'Left':
            self.setheading(180)
            self.old_xy = self.pos()
            self._position = (self.xcor() - self.game.move_step, self.ycor())

        elif direction == 'Up':
            self.setheading(90)
            self.old_xy = self.pos()
            self._position = (self.xcor(), self.ycor() + self.game.move_step)

        snake_dead = False
        is_clear_to_move = True
        for o in self.objects:

            # check for collision with border or body
            if (o.name == 'Wall' or o.name == 'Body') and o.pos() == self.pos():
                # snake is dead

                # reset snake
                self.reset()
                is_clear_to_move = False
                snake_dead = True
                break

            # snake eated food? If yes increase if not move
            elif o.name == 'Food' and o.pos() == self.pos():

                self.food_count += 1
                if self.food_count_record <= self.food_count:
                    self.food_count_record = self.food_count
                # snake increase
                self.grow()
                self.wn.update()
                is_clear_to_move = False

                # move food to another coordinates
                o.place_food()

        if is_clear_to_move:
            self.move()

        if self.record and snake_dead == False:
            # write dictionary of inputs to csv file
            logRecordToFile(inputs)
            self.records_count += 1

        # if self.model_loaded:
            # self.scaned_inputs = self.scan()

        # updateScore()
        turtle.clear()
        turtle.penup()
        turtle.hideturtle()

        # write to window eated food
        turtle.setposition(0, 300)
        turtle.write(F"Foud food: {self.food_count}, best {self.food_count_record}", False,
                     align="center", font=("Arial", 30, 'normal'))

        # write to window made records
        turtle.setposition(-350, -10)
        turtle.write(self.records_count, False, align="center",
                     font=("Arial", 12, 'normal'))

        self.wn.update()
        # g.log_data()

    # reset snake to starting possition

    def reset(self):
        # reset snake's parameters
        self.setpos(0, 0)
        self.setheading(90)
        self.food_count = 0

        # remove all tail objects
        for b in self.body:
            b.reset()
            b.penup()
            # turtles can not be deleted, they moved outside
            b.setpos(10000, 10000)
            b.hideturtle()
        self.body.clear()
        check_again = True
        while check_again == True:

            for b in self.objects:
                if b.name == 'Body':
                    self.objects.remove(b)
                    check_again = True
                    break
                else:
                    check_again = False

                # move food to another coordinates
                if b.name == 'Food':
                    b.place_food()
        self.grow()  # TODO uodega atsiranda ne prie kuno

    def grow(self):
        # tail.hideturtle()
        cube = Turtle()
        cube.name = 'Body'
        cube.shape('circle')
        cube.color('brown')
        cube.penup()
        cube.speed(0)
        cube.old_xy = [0, 0]

        cube.setposition(self.old_xy)  # set snake head pos to new cube
        self.body.insert(0, cube)  # insert new cube it first tail list pos
        self.objects.append(cube)

    def move(self):
        if len(self.body) > 0:

            self.body[-1].setposition(self.old_xy)
            self.body.insert(0, self.body[-1])
            self.body.pop(-1)

    # snake scaner

    def scan(self, showDetectsInWindow=True):
        # define directions of scan
        directions = [
            ["Right", 1, 0],
            ["RightDown", 1, -1],
            ["Down", 0, -1],
            ["LeftDown", -1, -1],
            ["Left", -1, 0],
            ["UpLeft", -1, 1],
            ["Up", 0, 1],
            ["UpRight", 1, 1]
        ]
        # temp list with wall, food and body distances
        founded_objects = []

        for direction in directions:
            x, y = self.pos()
            xdir = direction[1]
            ydir = direction[2]
            found_body = False
            found_wall = False
            while found_wall == False:

                x += xdir * self.game.move_step
                y += ydir * self.game.move_step

                for _object in self.objects:
                    if _object.pos() == (x, y):
                        dist = self.distance(_object)/self.game.move_step
                        if _object.name == 'Body' and found_body != False:
                            continue
                        if _object.name == 'Body' and found_body == False:
                            found_body = True
                        founded_objects.append(
                            [direction[0] + '_' + _object.name, dist, _object.pos()])
                        if _object.name == 'Wall':
                            found_wall = True
                            break

        # convert scaner data to inputs
        inputs = {
            'Right_Food': 0,
            'Right_Body': 0,
            'Right_Wall': 0,
            'RightDown_Food': 0,
            'RightDown_Body': 0,
            'RightDown_Wall': 0,
            'Down_Food': 0,
            'Down_Body': 0,
            'Down_Wall': 0,
            'LeftDown_Food': 0,
            'LeftDown_Body': 0,
            'LeftDown_Wall': 0,
            'Left_Food': 0,
            'Left_Body': 0,
            'Left_Wall': 0,
            'UpLeft_Food': 0,
            'UpLeft_Body': 0,
            'UpLeft_Wall': 0,
            'Up_Food': 0,
            'Up_Body': 0,
            'Up_Wall': 0,
            'UpRight_Food': 0,
            'UpRight_Body': 0,
            'UpRight_Wall': 0,
            'Dist_to_Food': 0,
            'Body_Lenght': 0,
            'Direction': 0

        }
        for s in founded_objects:
            # fill scans data to inputs dictionary
            inputs[s[0]] = s[1]

            # show detects in window
            if showDetectsInWindow:
                write(s[2], s[0], 'red')

        # detect dist to food
        for obj in self.objects:
            if obj.name == 'Food':
                dist = self.distance(obj)/self.game.move_step
                inputs['Dist_to_Food'] = dist

                # show food dist in window
                if showDetectsInWindow:
                    write(obj.pos(), f'Dist. to food: {dist}', 'blue')
                    break

        # add tail = body lenght to inputs
        body_lenght = len(self.body)
        inputs['Body_Lenght'] = body_lenght

        # show detects in window
        if showDetectsInWindow:
            write(self.pos(), f'Body lenght: {body_lenght}', 'blue')

        return inputs
