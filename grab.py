from PIL import ImageGrab

if __name__ == "__main__":
    #ImageGrab.grab(bbox=(1920,0,3600,1080)).show()
    up_left_x = 1920 + 550
    bottom_right_x = 3600 - 500

    up_left_y = 150
    bottom_right_y = 300

    ImageGrab.grab(bbox=(up_left_x, up_left_y,
                         bottom_right_x, bottom_right_y)).show()