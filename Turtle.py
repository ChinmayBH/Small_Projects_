#### Remove comment for design you want #####
###To draw a simple square###
# import turtle

# bob = turtle.Turtle()
# bob.color("red","cyan")

# bob.begin_fill()
# bob.forward(100)
# bob.left(90)
# bob.forward(100)
# bob.left(90)
# bob.forward(100)
# bob.left(90)
# bob.forward(100)
# bob.end_fill()

# bob.penup()
# bob.fd(15)
# bob.pendown()

# bob.begin_fill()
# bob.forward(100)
# bob.left(90)
# bob.forward(100)
# bob.left(90)
# bob.forward(100)
# bob.left(90)
# bob.forward(100)
# bob.end_fill()

# turtle.done()
############################################################
###To draw star shape###
# import turtle


# star = turtle.Turtle()
# star.color("blue","yellow")
# star.speed(10)
# star.begin_fill()

# for i in range(200):
#   star.forward(300)
#   star.left(168.7)  
# star.end_fill() 
# turtle.done()
########################################################
# import turtle
# import math
# vis = turtle.Turtle()
# vis.speed(10)
# vis.color("Red","yellow")
# vis.begin_fill()
# for i in range(1000):
#     vis.forward(math.sqrt(i))
#     vis.left(i%180)
# vis.end_fill()
# turtle.done()
#######################################################
#####  Circular drifts like cycloid##########
# import turtle
# p = turtle.Turtle()
# p.getscreen().bgcolor("#994444")
# def star(turtle,size):
#     if size <= 10:
#         return
#     else:
#         for i in range(5):
#             turtle.forward(size)
#             star(turtle,size/2)
#             turtle.left(216)

# star(p,100)
# turtle.done()
######################################################
###################CORONA###########################
import turtle
t = turtle.Turtle()
s = turtle.Screen()
s.bgcolor("black")
t.pencolor("white")
a = 0
b = 0
t.speed(0)
t.penup()
t.goto(0,200)
t.pendown()
while True:
    t.forward(a)
    t.right(b)
    a+=3
    b+=1
    if b == 200:
        break
    t.hideturtle()
turtle.done()