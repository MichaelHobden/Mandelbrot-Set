import pygame
import numpy as np

def start_visualisation() -> None:
    """
    Initializes and runs the Mandelbrot set visualization using Pygame.

    This function sets up the initial window, handles user input (zooming and panning), 
    calculates the Mandelbrot set, and continuously updates the display with 
    the latest visualization. It adjusts the iteration count based on the 
    Laplacian variance and allows the user to interact with the fractal through 
    mouse scrolling and keyboard arrow keys.

    Returns:
        None
    """
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
    width, height = 1000, 1000
    max_iterations = 100
    
    v_min, v_max = 100, 200
    
    aspect_ratio = width / height
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if aspect_ratio > 1:
        updated_y_range = x_range / aspect_ratio
        y_mid = (y_max + y_min) / 2
        y_min = y_mid - (updated_y_range / 2)
        y_max = y_mid + (updated_y_range / 2)
        
    elif aspect_ratio < 1:
        updated_x_range = y_range * aspect_ratio
        x_mid = (x_max + x_min) / 2
        x_min = x_mid - (updated_x_range / 2)
        x_max = x_mid + (updated_x_range / 2)
    
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Mandelbrot Set Visualisation')
    
    font = pygame.font.Font(None, 36)  
    text_color = (255, 255, 255)
     
    clock = pygame.time.Clock()
    dt = 0
    
    running = True
    
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEWHEEL:
                x_position, y_position = pygame.mouse.get_pos()
                if event.y == 1:
                    zoom_factor = 0.95
                    
                    x_ratio = width / (x_max - x_min)
                    x_position_relative = x_min + (x_position / x_ratio)
                    y_ratio = height / (y_max - y_min)
                    y_position_relative = y_min + (y_position / y_ratio)
         
                    x_relative_left = (x_position_relative - x_min) / (x_max - x_min)
                    x_relative_right = 1 - x_relative_left 
                    y_relative_top = (y_position_relative - y_min) / (y_max - y_min)
                    y_relative_bottom = 1 - y_relative_top
                    
                    x_relative_left_scaled = x_relative_left * zoom_factor
                    x_relative_right_scaled = x_relative_right * zoom_factor
                    y_relative_top_scaled = y_relative_top * zoom_factor
                    y_relative_bottom_scaled = y_relative_bottom * zoom_factor
                    
                    x_min = x_position_relative - (x_relative_left_scaled * (x_max - x_min))
                    x_max = x_position_relative + (x_relative_right_scaled * (x_max - x_min))
                    y_min = y_position_relative - (y_relative_top_scaled * (y_max - y_min))
                    y_max = y_position_relative + (y_relative_bottom_scaled * (y_max - y_min))
                     
                if event.y == -1:
                    zoom_factor = 1/0.95
                    
                    x_ratio = width / (x_max - x_min)
                    x_position_relative = x_min + (x_position / x_ratio)
                    y_ratio = height / (y_max - y_min)
                    y_position_relative = y_min + (y_position / y_ratio)
         
                    x_relative_left = (x_position_relative - x_min) / (x_max - x_min)
                    x_relative_right = 1 - x_relative_left 
                    y_relative_top = (y_position_relative - y_min) / (y_max - y_min)
                    y_relative_bottom = 1 - y_relative_top
                    
                    x_relative_left_scaled = x_relative_left * zoom_factor
                    x_relative_right_scaled = x_relative_right * zoom_factor
                    y_relative_top_scaled = y_relative_top * zoom_factor
                    y_relative_bottom_scaled = y_relative_bottom * zoom_factor
                    
                    x_min = x_position_relative - (x_relative_left_scaled * (x_max - x_min))
                    x_max = x_position_relative + (x_relative_right_scaled * (x_max - x_min))
                    y_min = y_position_relative - (y_relative_top_scaled * (y_max - y_min))
                    y_max = y_position_relative + (y_relative_bottom_scaled * (y_max - y_min))
            elif event.type == pygame.KEYDOWN:
                movement_factor = 0.1
                if event.key == pygame.K_LEFT:
                    x_height_scaled = x_max - x_min
                    up_movement = x_height_scaled * movement_factor
                    x_max = x_max - up_movement
                    x_min = x_min - up_movement
                if event.key == pygame.K_RIGHT:
                    x_height_scaled = x_max - x_min
                    down_movement = x_height_scaled * movement_factor
                    x_max = x_max + down_movement
                    x_min = x_min + down_movement
                if event.key == pygame.K_UP:
                    y_height_scaled = y_max - y_min
                    left_movement = y_height_scaled * movement_factor
                    y_max = y_max - left_movement
                    y_min = y_min - left_movement
                if event.key == pygame.K_DOWN:
                    y_height_scaled = y_max - y_min
                    right_movement = y_height_scaled * movement_factor
                    y_max = y_max + right_movement
                    y_min = y_min + right_movement
            elif event.type == pygame.QUIT:
                running = False
        
        mandelbrot_set = create_mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iterations)
        mandelbrot_set_colored = convert_iterations_to_color(mandelbrot_set, max_iterations)
        laplacian_variance = calculate_laplacian_variance(mandelbrot_set)
        
        if laplacian_variance < v_min:
            max_iterations = round(max_iterations * 1.1)
        elif laplacian_variance > v_max:
            max_iterations = round(max_iterations / 1.1)
        
        screen.fill('white')
        
        for y in range(height):
            for x in range(width):
                screen.set_at((x, y), mandelbrot_set_colored[y, x])
                
        dt = clock.tick(30) / 1000   
        fps = round(1/dt, 2)
        
        fps = font.render(f'FPS: {fps}', True, text_color)
        screen.blit(fps, (10, 10))
        
        variance = font.render(f'L-Var: {laplacian_variance}', True, text_color)
        screen.blit(variance, (10, 30))

        pygame.display.flip()

    pygame.quit()

def create_mandelbrot_set(x_min: float, x_max: float, y_min: float, y_max: float, width: int, height: int, max_iterations: int) -> np.ndarray:
    """
    Generates a Mandelbrot set based on a defined region in the complex plane.

    Args:
        x_min (float): The minimum x-value in the complex plane.
        x_max (float): The maximum x-value in the complex plane.
        y_min (float): The minimum y-value in the complex plane.
        y_max (float): The maximum y-value in the complex plane.
        width (int): The width of the output image.
        height (int): The height of the output image.
        max_iterations (int): The maximum number of iterations for each point.

    Returns:
        np.ndarray: A 2D array of iteration counts for each pixel.
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    Z = np.zeros_like(C, dtype=complex)
    iterations = np.zeros_like(C, dtype=int)

    for i in range(max_iterations):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        iterations[mask] += 1
        
    return iterations

def convert_iterations_to_color(iterations: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Converts iteration counts from a Mandelbrot set to RGB colors.

    Args:
        iterations (np.ndarray): The array of iteration counts for each point.
        max_iterations (int): The maximum iteration count used for color scaling.

    Returns:
        np.ndarray: A 2D array representing RGB colors for each point in the Mandelbrot set.
    """
    t = iterations / max_iterations
    r = (9 * (1 - t) * t**3 * 255).astype(np.uint8)
    g = (15 * (1 - t)**2 * t**2 * 255).astype(np.uint8)
    b = (8.5 * (1 - t)**3 * t * 255).astype(np.uint8)
    
    mandelbrot_set = np.stack((r, g, b), axis=-1)
    return mandelbrot_set

def get_color(iteration: int, max_iterations: int) -> tuple[int, int, int]:
    """
    Returns an RGB color for a specific iteration count in the Mandelbrot set.

    Args:
        iteration (int): The iteration count for the point in the Mandelbrot set.
        max_iterations (int): The maximum iteration count for color scaling.

    Returns:
        tuple[int, int, int]: An RGB tuple representing the color for the given iteration.
    """
    if iteration == max_iterations:
        return (0, 0, 0) 
    else:
        t = iteration / max_iterations
        r = int(9 * (1 - t) * t**3 * 255)
        g = int(15 * (1 - t)**2 * t**2 * 255)
        b = int(8.5 * (1 - t)**3 * t * 255)
        return (r, g, b)

def calculate_laplacian_variance(array: np.ndarray, kernel: np.ndarray = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])) -> float:
    """
    Calculates the variance of the Laplacian of an input array.

    Args:
        array (np.ndarray): The 2D array representing the Mandelbrot set.
        kernel (np.ndarray, optional): The kernel for convolution, default is a Laplacian kernel.

    Returns:
        float: The variance of the Laplacian of the array.
    """
    width, height = array.shape
    array = np.pad(array, pad_width=1, mode='constant', constant_values=1)
    
    convolved_array = np.zeros((width, height))
    
    for i in range(width):
        for j in range(height):
            x = i+1
            y = j+1
            
            position_matrix = [
                [array[x-1, y-1], array[x, y-1], array[x+1, y-1]],
                [array[x-1, y],   array[x, y],   array[x+1, y]],
                [array[x-1, y+1], array[x, y+1], array[x+1, y+1]]]
            
            convolved_array[i, j] = np.sum(np.multiply(position_matrix, kernel))
            
    return np.var(convolved_array)

if __name__ == '__main__':
    start_visualisation()
