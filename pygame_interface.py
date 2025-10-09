# pygame_interface.py
"""
Pygame visualization for Tic-Tac-Toe AI comparison.
Provides real-time visualization of algorithm decision making.
"""

import pygame
import sys
import time
from game_engine import TicTacToe
from ai_algorithms import BFSAI, GreedyAI, MinimaxAI

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BOARD_SIZE = 400
CELL_SIZE = BOARD_SIZE // 3
MARGIN = 50

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
BLUE = (100, 100, 255)
LIGHT_BLUE = (200, 200, 255)
YELLOW = (255, 255, 100)

class PygameVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe AI Comparison")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game components
        self.game = TicTacToe()
        self.bfs_ai = BFSAI(self.game)
        self.greedy_ai = GreedyAI(self.game)
        self.minimax_ai = MinimaxAI(self.game)
        
        # Visualization state
        self.current_ai = None
        self.visualization_data = {
            'current_algorithm': None,
            'nodes_evaluated': 0,
            'computation_time': 0,
            'search_tree': [],  # For visualizing explored nodes
            'current_move': None
        }
        self.auto_play = False
        self.game_speed = 1.0  # Seconds between AI moves
    
    def draw_board(self):
        """Draw the Tic-Tac-Toe board."""
        board_x = MARGIN
        board_y = MARGIN
        
        # Draw board background
        pygame.draw.rect(self.screen, WHITE, (board_x, board_y, BOARD_SIZE, BOARD_SIZE))
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(self.screen, BLACK, 
                           (board_x + i * CELL_SIZE, board_y),
                           (board_x + i * CELL_SIZE, board_y + BOARD_SIZE), 3)
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK,
                           (board_x, board_y + i * CELL_SIZE),
                           (board_x + BOARD_SIZE, board_y + i * CELL_SIZE), 3)
        
        # Draw X's and O's
        for i in range(9):
            row = i // 3
            col = i % 3
            x = board_x + col * CELL_SIZE + CELL_SIZE // 2
            y = board_y + row * CELL_SIZE + CELL_SIZE // 2
            
            if self.game.board[i] == 'X':
                # Draw X
                pygame.draw.line(self.screen, RED, 
                               (x - 30, y - 30), (x + 30, y + 30), 4)
                pygame.draw.line(self.screen, RED,
                               (x + 30, y - 30), (x - 30, y + 30), 4)
            elif self.game.board[i] == 'O':
                # Draw O
                pygame.draw.circle(self.screen, BLUE, (x, y), 30, 4)
        
        # Highlight current move suggestion
        if self.visualization_data['current_move'] is not None:
            move = self.visualization_data['current_move']
            row = move // 3
            col = move % 3
            highlight_x = board_x + col * CELL_SIZE
            highlight_y = board_y + row * CELL_SIZE
            pygame.draw.rect(self.screen, YELLOW, 
                           (highlight_x + 5, highlight_y + 5, 
                            CELL_SIZE - 10, CELL_SIZE - 10), 3)
    
    def draw_algorithm_info(self):
        """Draw algorithm selection and information panel."""
        panel_x = BOARD_SIZE + MARGIN * 2
        panel_y = MARGIN
        panel_width = SCREEN_WIDTH - panel_x - MARGIN
        panel_height = 200
        
        # Draw panel background
        pygame.draw.rect(self.screen, LIGHT_BLUE, 
                       (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, BLACK, 
                       (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font.render("AI Algorithm Controller", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Algorithm buttons
        algorithms = [
            ("Breadth-First Search", self.bfs_ai),
            ("Greedy Best-First", self.greedy_ai),
            ("Minimax Alpha-Beta", self.minimax_ai)
        ]
        
        button_width = 250
        button_height = 40
        button_spacing = 10
        
        for i, (name, ai) in enumerate(algorithms):
            button_y = panel_y + 60 + i * (button_height + button_spacing)
            color = GREEN if self.current_ai == ai else GRAY
            
            # Draw button
            pygame.draw.rect(self.screen, color, 
                           (panel_x + 10, button_y, button_width, button_height))
            pygame.draw.rect(self.screen, BLACK, 
                           (panel_x + 10, button_y, button_width, button_height), 2)
            
            # Button text
            text = self.small_font.render(name, True, BLACK)
            self.screen.blit(text, (panel_x + 20, button_y + 12))
    
    def draw_metrics(self):
        """Draw real-time performance metrics."""
        metrics_x = MARGIN
        metrics_y = BOARD_SIZE + MARGIN * 2
        metrics_width = SCREEN_WIDTH - MARGIN * 2
        metrics_height = 150
        
        # Draw metrics panel
        pygame.draw.rect(self.screen, LIGHT_BLUE, 
                       (metrics_x, metrics_y, metrics_width, metrics_height))
        pygame.draw.rect(self.screen, BLACK, 
                       (metrics_x, metrics_y, metrics_width, metrics_height), 2)
        
        # Metrics title
        title = self.font.render("Algorithm Performance", True, BLACK)
        self.screen.blit(title, (metrics_x + 10, metrics_y + 10))
        
        # Current algorithm
        algo_name = "None" if self.current_ai is None else self.current_ai.__class__.__name__
        algo_text = self.small_font.render(f"Current Algorithm: {algo_name}", True, BLACK)
        self.screen.blit(algo_text, (metrics_x + 20, metrics_y + 50))
        
        # Performance metrics
        nodes_text = self.small_font.render(
            f"Nodes Evaluated: {self.visualization_data['nodes_evaluated']:,}", True, BLACK)
        time_text = self.small_font.render(
            f"Computation Time: {self.visualization_data['computation_time']:.4f}s", True, BLACK)
        
        self.screen.blit(nodes_text, (metrics_x + 20, metrics_y + 80))
        self.screen.blit(time_text, (metrics_x + 20, metrics_y + 110))
    
    def draw_game_info(self):
        """Draw game state information."""
        info_x = MARGIN
        info_y = BOARD_SIZE + MARGIN * 2 + 160
        
        # Game status
        status = "Game Over - " if self.game.game_over else ""
        if self.game.winner:
            status += f"Winner: {self.game.winner}"
        elif self.game.game_over:
            status += "Draw"
        else:
            status += f"Current Player: {self.game.current_player}"
        
        status_text = self.font.render(status, True, BLACK)
        self.screen.blit(status_text, (info_x, info_y))
        
        # Controls info
        controls = [
            "Controls: Click board to make move",
            "1-3: Select AI algorithm",
            "Space: Toggle auto-play",
            "R: Reset game"
        ]
        
        for i, control in enumerate(controls):
            control_text = self.small_font.render(control, True, BLACK)
            self.screen.blit(control_text, (info_x, info_y + 40 + i * 25))
    
    def handle_click(self, pos):
        """Handle mouse clicks for board moves and algorithm selection."""
        x, y = pos
        
        # Check if click is on the board
        if (MARGIN <= x <= MARGIN + BOARD_SIZE and 
            MARGIN <= y <= MARGIN + BOARD_SIZE):
            # Convert click to board position
            col = (x - MARGIN) // CELL_SIZE
            row = (y - MARGIN) // CELL_SIZE
            position = row * 3 + col
            
            # Make human move
            if not self.game.game_over and self.current_ai is None:
                self.game.make_move(position)
                
                # If game continues, let AI make a move
                if not self.game.game_over and self.current_ai is not None:
                    self.make_ai_move()
        
        # Check algorithm selection buttons
        panel_x = BOARD_SIZE + MARGIN * 2
        for i in range(3):
            button_y = MARGIN + 60 + i * 50
            if (panel_x + 10 <= x <= panel_x + 260 and 
                button_y <= y <= button_y + 40):
                algorithms = [self.bfs_ai, self.greedy_ai, self.minimax_ai]
                self.current_ai = algorithms[i]
                print(f"Selected {algorithms[i].__class__.__name__}")
    
    def make_ai_move(self):
        """Make a move using the current selected AI."""
        if self.current_ai and not self.game.game_over:
            move, nodes, comp_time = self.current_ai.find_move()
            
            if move is not None:
                self.visualization_data.update({
                    'current_algorithm': self.current_ai.__class__.__name__,
                    'nodes_evaluated': nodes,
                    'computation_time': comp_time,
                    'current_move': move
                })
                
                self.game.make_move(move)
                return True
        return False
    
    def run(self):
        """Main game loop."""
        running = True
        last_ai_move_time = 0
        
        while running:
            current_time = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.game.reset_game()
                        self.visualization_data['current_move'] = None
                    elif event.key == pygame.K_SPACE:  # Toggle auto-play
                        self.auto_play = not self.auto_play
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:  # Select AI
                        algorithms = [self.bfs_ai, self.greedy_ai, self.minimax_ai]
                        self.current_ai = algorithms[event.key - pygame.K_1]
            
            # Auto-play mode
            if (self.auto_play and self.current_ai and not self.game.game_over and
                current_time - last_ai_move_time > self.game_speed):
                if self.make_ai_move():
                    last_ai_move_time = current_time
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_algorithm_info()
            self.draw_metrics()
            self.draw_game_info()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        sys.exit()

# Main execution
if __name__ == "__main__":
    print("Starting Pygame Visualization...")
    print("Controls:")
    print("- Click on board to make moves")
    print("- Click algorithm buttons to select AI")
    print("- Press 1, 2, 3 to quickly select AI")
    print("- Space: Toggle auto-play")
    print("- R: Reset game")
    print()
    
    visualizer = PygameVisualizer()
    visualizer.run()