
from pynput import keyboard





def on_press(key):
    global break_program
    print(key)
    if key == keyboard.Key.end:
        break_program = True
        return False



