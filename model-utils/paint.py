#Borrowed (somewhat) from https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06
from tkinter import *
from tkinter import filedialog
from tkinter.colorchooser import askcolor
import os
import io
import PIL.Image
import subprocess
import time

def create_filename():
    """Makes a filename that isn't a duplicate in a given directory"""
    name = 19
    while os.path.isfile('test/' + str(name) + '.jpg') is True:
        name += 1;
    filename = str(name) + '.jpg'
    return filename

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def save_popup(self, filename):
        """Creates a popup confirming that you have saved"""
        toplevel = Toplevel()
        label1 = Label(toplevel, text=filename + " has been saved!", height=0, width=30)
        label1.pack()

    def __init__(self):
        """Initializes the Canvas and drawing board"""
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.save_button = Button(self.root, text='save', command=self.save)
        self.save_button.grid(row=0, column=4)

        self.clear_button = Button(self.root, text='clear', command=self.delete_all)
        self.clear_button.grid(row=0, column=5)

        self.toggle_size = Button(self.root, text='Size: 500x500', command=self.toggle_sizes)
        self.toggle_size.grid(row=0, column=6)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.set(5)
        self.choose_size_button.grid(row=0, column=7)

        self.c = Canvas(self.root, bg='white', width=492, height=492)
        self.c.grid(row=1, columnspan=8)

        self.setup()
        self.root.mainloop()

    def setup(self):
        """Second Initialization of Non Canvas related variables"""
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.smaller_size = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        """Toggles the pen"""
        self.activate_button(self.pen_button)
        self.choose_size_button.set(5)

    def use_brush(self):
        """Toggles Brush"""
        self.activate_button(self.brush_button)
        self.choose_size_button.set(5)

    def choose_color(self):
        """Choose any color for the pen"""
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def color_change(self):
        "Change the color of the button when depressed"

    def toggle_sizes(self):
        """Toggles between the sizes 200 by 200 for training data and 500 by 500 for testing data"""
        self.smaller_size ^= True
        if self.smaller_size:
            self.toggle_size.config(text="Size: 200x200")
        else:
            self.toggle_size.config(text="Size: 500x500")

    def use_eraser(self):
        """Toggles the eraser"""
        self.activate_button(self.eraser_button, eraser_mode=True)
        self.choose_size_button.set(10)

    def activate_button(self, some_button, eraser_mode=False):
        """Helps activate the pen, eraser, and brush"""
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        """Create lines every second where your pointer is. This creates a smoother painting experience"""
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        """Resets the old line start points"""
        self.old_x, self.old_y = None, None

    def save(self):
        """Saves the canvas as a JPEG using Ghostscript and PIL"""
        filename = create_filename()
        ps = self.c.postscript(colormode='color')
        im = PIL.Image.open(io.BytesIO(ps.encode('utf-8')))
        if self.smaller_size:
            im = im.resize((200, 200))
        else:
            im = im.resize((500, 500))
        im.save('test/' + filename, 'jpeg')
        self.save_popup(filename)
    
    def delete_all(self):
        """Clears the canvas and drawing board"""
        self.c.delete("all")


if __name__ == '__main__':
    if not os.path.exists('test'):
        os.makedirs('test')
    Paint()