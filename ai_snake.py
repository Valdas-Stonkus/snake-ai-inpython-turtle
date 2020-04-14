# Snake game with AI in Python3.8 By Valdas Stonkus
import turtle
from functions_of_game import *
from wall import Wall
from snake import Snake
from food import Food

# create config


class ConfigGame:
    version = 'v1.2'
    win_title = 'Snake game with AI in Python by @Valdas Stonkus'
    win_width = 1000
    win_height = 800
    window_bg_color = 'white'
    move_step = 30  # this is also size of one cell in grid
    grid_size = 15  # 3=3x3 it can be only odd numbers: 3, 5, 7, 9 etc.
    train_model = None
    train_scaler = None

    # calculations of variables
    max_grid_size = grid_size + 2  # grid with walls

    # calculate first field cell possition of left bottom
    first_cell_pos_y = -(grid_size * move_step / 2 + move_step/2)
    first_cell_pos_x = first_cell_pos_y

    grid_dist = grid_size * move_step  # default 450
    # collums = grid_dist / move_step
    max_dist = (grid_dist / 2) - (move_step / 2)  # maximum grids distances
    # delay = 0


g = ConfigGame()

# set window of game
wn = turtle.Screen()
wn.title(f'{g.win_title} {g.version}')
wn.bgcolor(g.window_bg_color)
wn.setup(width=g.win_width, height=g.win_height)
wn.tracer(0)  # Turns off screen updates


# draw grid
draw_grid(g)
wn.update()

# create matrix cords_map ex.: get_cords((0,0)) - returns [-60, -60]
get_cords = create_cords_map(g)

# create game_map -------------------------------------------------------
game_map = {}  # there we store all objects
object_map = []

# Insert walls in coordinates to 0,0 ... 0,4 etc arround grid square
# insert horizontals walls
y = g.max_grid_size-1
for x in range(g.max_grid_size):

    object_map.append(Wall((x, 0), g))  # horizontall bottom
    object_map.append(Wall((x, y), g))


# insert verticals walls
x = g.max_grid_size-1
for y in range(1, g.max_grid_size-1):
    object_map.append(Wall((0, y), g))
    object_map.append(Wall((x, y), g))

# -----------------------------------------------------------------------

# create snake
snake = Snake('Adam', (0, 0), g, wn, game_map, object_map, 180)

# create food
food = Food(g, snake, wn, game_map, get_cords)
# add food object to map
object_map.append(food)


# Detect mouse possition


def mouseClick(x, y):
    print(x, y)  # print mouse clik cords

    # check mouse coordinates of clicks. If button "Record" was clicked"
    if x > -475 and x < -447 and y > -15 and y < 15:

        placeButton('square', 'red', (-460, 0))
        wn.update()
        snake.record = True
        print('Record started')

    # check mouse coordinates of clicks. If button "Stop" was clicked"
    if x > -475 and x < -447 and y > -65 and y < -35:

        placeButton('square', 'green', (-460, 0))
        wn.update()
        snake.record = False
        print('Record stoped')

    # check mouse coordinates of clicks.
    if x > -475 and x < -447 and y > -115 and y < -55:

        loadTrainedModel(g)
        # wn.update()


# listen mouse clicks
turtle.onscreenclick(mouseClick, 1)

# draw button "Record"
placeButton('square', 'green', (-460, 0))
placeText('Record', (-400, -10))

# draw  button "Stop"
placeButton('square', 'black', (-460, -50))
placeText('Stop', (-400, -60))

# draw button load trained Neuron model
placeButton('square', 'brown', (-460, -100))
placeText('Load snake model', (-340, -110))


# draw button "Auto Snake play"

# draw button "Stop Auto Snake play"


def prepareAndPredict():
    if not g.train_model and not g.train_scaler:
        loadTrainedModel(g)
    inputs = snake.scan(False)
    prediction = predict(inputs, g)
    return prediction


def autoPlay():

    for _ in range(10000):

        pred = prepareAndPredict()
        if pred == 'Up':
            snake.go_up()
        elif pred == 'Down':
            snake.go_down()
        elif pred == 'Left':
            snake.go_left()
        elif pred == 'Right':
            snake.go_right()


# Keyboard bindings, make snake move by keypress
wn.onkeypress(snake.go_up, 'Up')
wn.onkeypress(snake.go_down, 'Down')
wn.onkeypress(snake.go_left, 'Left')
wn.onkeypress(snake.go_right, 'Right')
wn.onkeypress(snake.scan, 's')  # scan

wn.onkeypress(trainSnake, 't')
wn.onkeypress(checkTrainedModel, 'c')
wn.onkeypress(prepareAndPredict, 'p')

wn.onkeypress(autoPlay, 'a')


wn.listen()
wn.update()
wn.mainloop()
