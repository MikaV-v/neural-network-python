from tkinter import *
import cv2, os
ans_list=['Man','Man',"woman","Man"]
most_common=0
def reset(event):
    global ans_list, status, most_common
    status = 0
    spians_set = set(ans_list)
    most_common = None
    qty_most_common = 0
    for item in spians_set:
        qty = ans_list.count(item)
        if qty > qty_most_common:
            qty_most_common = qty
            most_common = item
    print(most_common)
    text.insert(1.0, most_common)
    ans_list = []
    status = 0
    text.get('1.0', 'end')
    text.tag_add('title', 1.0, '1.end')
    text.tag_config('title', font=("Verdana", 60, 'bold'), justify=CENTER)
    text.pack()

def start(event):
    global status
    text.delete(1.0, END)
    status = 1

root = Tk()

text = Text(width=50, height=40)
text.pack()



Button_status_reset = Button(root, text = 'Show Answer', width=40, height=30)
Button_status_start = Button(root, text = 'start', width=40, height=30)

Button_status_start.bind('<Button-1>', start)
Button_status_start.bind('<Return>', start)
Button_status_start.pack()

Button_status_reset.bind('<Button-1>', reset)
Button_status_reset.bind('<Return>', reset)
Button_status_reset.pack()

root.mainloop()