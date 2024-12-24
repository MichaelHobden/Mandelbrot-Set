# Mandelbrot Set Visualizer

This project is a Python-based visualizer for the Mandelbrot set using the `pygame` library and `numpy` for handling numerical computations. The program generates and displays the Mandelbrot set, allowing zooming and movement using the mouse and keyboard. It also adjusts the maximum iterations dynamically based on the Laplacian variance.

## Features

- **Mandelbrot Set Visualization**: Renders the Mandelbrot set in a graphical window using `pygame`.
- **Zooming**: Zoom in or out using the mouse scroll wheel.
- **Panning**: Move the view by pressing the arrow keys.
- **Dynamic Adjustments**: The program dynamically adjusts the maximum number of iterations based on the Laplacian variance of the current view.
- **FPS Display**: Shows the current frames per second in the corner of the window.
- **Laplacian Variance**: Displays the calculated Laplacian variance to indicate the smoothness of the set.

## Requirements

- Python 3.x
- `pygame` library
- `numpy` library

You can install the necessary libraries with the following commands:

```python
pip install pygame numpy
```

## Usage

To run the program, simply execute the script in your terminal:

```bash
python mandelbrot_visualizer.py
```

- **Zooming**: Use the mouse scroll wheel to zoom in and out of the fractal.
- **Panning**: Hold the right mouse button and drag to move around the fractal.
- **Key Controls**:
  - Arrow keys to pan the view.
  - `+` and `-` keys to increase or decrease the number of iterations.
  - `Esc` to quit the program.

## Code Description

### `start_visualisation()`

The main function that initializes the `pygame` window, sets up the game loop, handles user input (mouse and keyboard), and renders the Mandelbrot set on the screen. It dynamically adjusts the zoom level and the view area based on user input.

### `create_mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iterations)`

Generates the Mandelbrot set for the specified area in the complex plane. The function returns an array of iterations for each point in the plane.

### `convert_iterations_to_color(iterations, max_iterations)`

Converts the number of iterations for each point into a color. The color is determined based on the iteration count using a smooth color gradient.

### `get_color(iteration, max_iterations)`

Helper function to calculate the color for a single iteration point based on the number of iterations.

### `calculate_laplacian_variance(array, kernel)`

Calculates the Laplacian variance of the Mandelbrot set to determine the smoothness of the fractal. This value is used to adjust the maximum number of iterations dynamically during visualization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Mandelbrot set algorithm is well-known in mathematics and fractals.
- `pygame` and `numpy` are open-source libraries that made this project possible.
