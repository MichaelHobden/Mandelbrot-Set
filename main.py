import matplotlib.pyplot as plt
import numpy as np
import pygame


def start_visualisation():
    
    # Create starting position of the visualisation
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
    width, height = 400, 400
    max_iterations = 100
    
    # This code is to adjust the aspect ratio
    # It is outside of the game loop as the movement is coded in a way that it won't affect aspect ratio
    
    aspect_ratio = width / height
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if aspect_ratio > 1:
        # Screen width greater than height
        updated_y_range = x_range / aspect_ratio
        
        y_mid = (y_max + y_min) / 2
        y_min = y_mid - (updated_y_range / 2)
        y_max = y_mid + (updated_y_range / 2)
        
    elif aspect_ratio < 1:
        # Screen height greater than width
        updated_x_range = y_range * aspect_ratio
        
        x_mid = (x_max + x_min) / 2
        x_min = x_mid - (updated_x_range / 2)
        x_max = x_mid + (updated_x_range / 2)
    
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Mandelbrot Set Visualisation')
    
    # Configuration for FPS Text
    font = pygame.font.Font(None, 36)  
    text_color = (255, 255, 255)
     
    # Code for game loop
    clock = pygame.time.Clock()
    dt = 0
    
    running = True
    
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEWHEEL:
                x_position, y_position = pygame.mouse.get_pos()
                if event.y == 1:
                    # Scroll Up: Recalculate values to zoom in 10%
                    zoom_factor = 0.95
                    
                    x_ratio = width / (x_max - x_min)
                    x_position_relative = x_min + (x_position / x_ratio)
                    y_ratio = height / (y_max - y_min)
                    y_position_relative = y_min + (y_position / y_ratio)
         
                    x_relative_left = (x_position_relative - x_min) / (x_max - x_min)
                    x_relative_right = 1 - x_relative_left 
                    y_relative_top = (y_position_relative - y_min) / (y_max - y_min)
                    y_relative_bottom = 1 - y_relative_top
                    
                    print(x_relative_left, x_relative_right)
                    print(y_relative_top, y_relative_bottom)
                    
                    x_relative_left_scaled = x_relative_left * zoom_factor
                    x_relative_right_scaled = x_relative_right * zoom_factor
                    y_relative_top_scaled = y_relative_top * zoom_factor
                    y_relative_bottom_scaled = y_relative_bottom * zoom_factor
                    
                    x_min = x_position_relative - (x_relative_left_scaled * (x_max - x_min))
                    x_max = x_position_relative + (x_relative_right_scaled * (x_max - x_min))
                    y_min = y_position_relative - (y_relative_top_scaled * (y_max - y_min))
                    y_max = y_position_relative + (y_relative_bottom_scaled * (y_max - y_min))
                     
                if event.y == -1:
                    # Scroll Down
                    # Scroll Up: Recalculate values to zoom in 10%
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
            elif event.type == pygame.QUIT:
                running = False
        
        # print(x_min, x_max, y_min, y_max)
               
        mandelbrot_set = create_mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iterations)
    
        
        screen.fill('white')
        
        for y in range(height):
            for x in range(width):
                screen.set_at((x, y), mandelbrot_set[y, x])
                
        dt = clock.tick(60) / 1000   
        fps = round(1/dt, 2)
        
        text = font.render(f'FPS: {fps}', True, text_color)
        screen.blit(text, (10, 10))

        # Update the display
        pygame.display.flip()
        
        


    pygame.quit()
        

def mandelbrot(c: complex, max_iterations: int) -> int:
    z = 0
    
    for i in range(max_iterations):
        if abs(z) > 2:
            return i
        else:
            z = z * z + c
    return max_iterations

def create_mandelbrot_set(x_min: int, x_max: int, y_min: int, y_max: int, width: int, height: int, max_iterations: int) -> np.array:
    '''Returns a mandelbrot set based off of a values that represent a box in the complex plane
    The set is represented by colors rather than iterations,  this is to reduce overhead when it comes to processing (Not sure if it does XD)
    '''
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    Z = np.zeros_like(C, dtype=complex)
    iterations = np.zeros_like(C, dtype=int)

    for i in range(max_iterations):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        iterations += mask
    
    mandelbrot_set = np.array([get_color(iter, max_iterations) for iter in iterations.ravel()])
    mandelbrot_set = mandelbrot_set.reshape((height, width, 3))
    
    return mandelbrot_set

def get_color(iteration, max_iterations):
    if iteration == max_iterations:
        return (0, 0, 0) 
    else:
        t = iteration / max_iterations
        r = int(9 * (1 - t) * t**3 * 255)
        g = int(15 * (1 - t)**2 * t**2 * 255)
        b = int(8.5 * (1 - t)**3 * t * 255)
        return (r, g, b)

if __name__ == '__main__':
    start_visualisation()