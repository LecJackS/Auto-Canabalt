#include <iostream>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cmath> // round()
//#include <cstring>
//#include <cstdlib>
//#include <gdk/gdk.h>
//#include <gtk/gtk.h>
//#include <X11/Xutil.h>
//#include </usr/include/gtk-3.0/gtk/gtk.h>
//#include <ncurses.h>

class BasicCoordinates
{
private:
    long long X;
    long long Y;

public:
    BasicCoordinates(const long long & _X, const long long & _Y)
    {
        X = _X;
        Y = _Y;
    }
    long long getX()
    {
        return X;
    }
    long long getY()
    {
        return Y;
    }

};

bool StringContainsInteger(const std::string & str)
// true : if the string contains an integer number (possibly starting with a sign)
// false: if the string contains some other character(s)
{
    std::string::size_type str_len = str.length();
    if (str_len == 0) return false;

    bool sign_present = (str[0] == '-' || str[0] == '+');
    if (str_len == 1 && sign_present) return false;

    for (std::string::size_type i = 0; i < str_len; i++)
    {
        if (i == 0 && sign_present) continue;
        if (! std::isdigit((unsigned char) str[i])) return false;
    }

    return true;
}


// https://stackoverflow.com/a/59663458/1997354
inline __attribute__((always_inline)) \
unsigned long MyXGetPixel(XImage * ximage, const int & x, const int & y)
{
    return (*ximage->f.get_pixel)(ximage, x, y);
}

XColor MyGetPixelColor(Display *my_display, int x_coord, int y_coord)
{
    XColor pixel_color;
    XImage *screen_image = XGetImage(
        my_display,
        XRootWindow(my_display, XDefaultScreen(my_display)),
        x_coord, y_coord,
        1, 1,
        AllPlanes,
        XYPixmap
    );
    (&pixel_color)->pixel = XGetPixel(screen_image, 0, 0);
    XFree(screen_image);
    XQueryColor(my_display, XDefaultColormap(my_display, XDefaultScreen(my_display)), &pixel_color);

    return pixel_color;
}




int main(const int argc, const char * argv[])
{

    const int given_arguments_count = argc - 1;
    if (given_arguments_count != 2)
    {
        std::cerr
            << "Fatal error occurred while checking\n"
            << "the number of given arguments\n"
            << "--------------------------------------\n"
            << "In the function : " << __func__ << std::endl
            << "At the command  : " << "given_arguments_count\n"
            << "Given arguments : " << given_arguments_count << std::endl
            << "Error message   : " << "This program is expecting exactly 2 arguments\n"
            << "                  being the X Y coordinates of a pixel on the screen\n";
        return 1;
    }

    const std::string cli_argument_1 = argv[1];
    if (! StringContainsInteger(cli_argument_1))
    {
        std::cerr
            << "Fatal error occurred while checking\n"
            << "if the first argument contains a number\n"
            << "----------------------------------------\n"
            << "In the function : " << __func__ << std::endl
            << "At the command  : " << "StringContainsInteger\n"
            << "Input string    : " << cli_argument_1 << std::endl
            << "Error message   : " << "The first argument is not an integer number\n";
        return 1;
    }

    const std::string cli_argument_2 = argv[2];
    if (! StringContainsInteger(cli_argument_2))
    {
        std::cerr
            << "Fatal error occurred while checking\n"
            << "if the second argument contains a number\n"
            << "----------------------------------------\n"
            << "In the function : " << __func__ << std::endl
            << "At the command  : " << "StringContainsInteger\n"
            << "Input string    : " << cli_argument_2 << std::endl
            << "Error message   : " << "The second argument is not an integer number\n";
        return 1;
    }

    long long x_coord;
    try
    {
        x_coord = std::stoll(cli_argument_1);
    }
    catch (const std::exception & input_exception)
    {
        std::cerr
            << "Fatal error occurred while converting\n"
            << "the first argument to an integer variable\n"
            << "-------------------------------------\n"
            << "In the function : " << __func__ << std::endl
            << "At the command  : " << input_exception.what() << std::endl
            << "Input string    : " << cli_argument_1 << std::endl
            << "Error message   : " << "The first number argument is too big an integer\n";
        return 1;
    }

    long long y_coord;
    try
    {
        y_coord = std::stoll(cli_argument_2);
    }
    catch (const std::exception & input_exception)
    {
        std::cerr
            << "Fatal error occurred while converting\n"
            << "the second argument to an integer variable\n"
            << "--------------------------------------\n"
            << "In the function : " << __func__ << std::endl
            << "At the command  : " << input_exception.what() << std::endl
            << "Input string    : " << cli_argument_2 << std::endl
            << "Error message   : " << "The second number argument is too big an integer\n";
        return 1;
    }


    BasicCoordinates * pixel_coords = new BasicCoordinates(x_coord, y_coord);
    std::cout << "X = " << pixel_coords->getX() << std::endl;
    std::cout << "Y = " << pixel_coords->getY() << std::endl << std::endl;


    Display * my_display = XOpenDisplay(NULL);
    Screen * my_screen = XDefaultScreenOfDisplay(my_display);

    const int screen_width = my_screen->width;
    const int screen_height = my_screen->height;

    if (x_coord >= screen_width)
    {
        std::cerr
            << "X coord bigger than the screen with\n"; // TEMP

        return 1;
    }

    if (y_coord >= screen_height)
    {
        std::cerr
            << "Y coord bigger than the screen height\n"; // TEMP

        return 1;
    }



    XWindowAttributes root_window_attributes;
    XGetWindowAttributes(my_display, DefaultRootWindow(my_display), & root_window_attributes);


    XColor screen_pixel_color;


    //MyGetPixelColor(my_display, x_coord, y_coord, & screen_pixel_color);
    screen_pixel_color = MyGetPixelColor(my_display, x_coord, y_coord);
    //GetPixelColor(my_display, x_coord, y_coord, & screen_pixel_color);

    XCloseDisplay(my_display);



    unsigned short raw_r_value = screen_pixel_color.red;
    unsigned short raw_g_value = screen_pixel_color.green;
    unsigned short raw_b_value = screen_pixel_color.blue;
/*
    std::cout
        << "Raw Values" << std::endl
        << "R = " << raw_r_value << std::endl
        << "G = " << raw_g_value << std::endl
        << "B = " << raw_b_value << std::endl;

    std::cout << std::endl;
*/
    std::cout
//            << "Normalized" << std::endl
        << "R = " << round(raw_r_value / 256.0) << std::endl
        << "G = " << round(raw_g_value / 256.0) << std::endl
        << "B = " << round(raw_b_value / 256.0) << std::endl;



    return 0;

}