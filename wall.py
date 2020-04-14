from turtle import Turtle
from functions_of_game import create_cords_map


class Wall(Turtle):
    def __init__(self, pos, g):
        super().__init__()
        get_cords = create_cords_map(g)
        self.shape('square')
        self.color('black')
        self.penup()
        self.speed(0)
        self.name = 'Wall'
        self.setpos(get_cords[pos])  # get cords from matrix
