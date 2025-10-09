# game_engine.py
"""
Core Tic-Tac-Toe game engine.
Handles game state, moves, win conditions, and board operations.
"""

class TicTacToe:
    def __init__(self):
        """Initialize a new game."""
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board = [' '] * 9  
        self.current_player = 'X'  
        self.game_over = False
        self.winner = None
        self.move_history = []  
    
    def make_move(self, position):
        """
        Make a move at the specified position.
        
        Args:
            position (int): Position on board (0-8)
            
        Returns:
            bool: True if move was valid and made, False otherwise
        """
        if self.game_over or not self.is_valid_move(position):
            return False
        
        # Make the move
        self.board[position] = self.current_player
        self.move_history.append((self.current_player, position))
        
        # Check for win or draw
        self._check_game_state()
        
        # Switch player if game continues
        if not self.game_over:
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return True
    
    def is_valid_move(self, position):
        """Check if a move is valid."""
        return 0 <= position <= 8 and self.board[position] == ' '
    
    def get_available_moves(self):
        """Get list of available move positions."""
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def _check_game_state(self):
        """Check if the game has been won or is a draw."""
        # Check for win
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns  
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combinations:
            if (self.board[combo[0]] == self.board[combo[1]] == 
                self.board[combo[2]] != ' '):
                self.winner = self.board[combo[0]]
                self.game_over = True
                return
        
        # Check for draw
        if not self.get_available_moves():
            self.game_over = True
            self.winner = None  # Draw
    
    def get_board_state(self):
        """Return current board state as a tuple (hashable for AI)."""
        return tuple(self.board)
    
    def get_game_status(self):
        """Return current game status summary."""
        return {
            'board': self.board[:],  
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'available_moves': self.get_available_moves()
        }
    
    def print_board(self):
        """Print the current board to console."""
        print("\n")
        for i in range(3):
            row = self.board[i*3:(i+1)*3]
            print(" " + " | ".join(row))
            if i < 2:
                print("-----------")
        print(f"Current player: {self.current_player}")
        if self.game_over:
            print(f"Game over! Winner: {self.winner if self.winner else 'Draw'}")


# Utility functions for AI algorithms
def get_board_hash(board):
    """Create a hashable representation of the board state."""
    return ''.join(board)

def get_opponent(player):
    """Get the opponent player symbol."""
    return 'O' if player == 'X' else 'X'


# Test the game engine
if __name__ == "__main__":
    # test game
    game = TicTacToe()
    
    print("Testing Tic-Tac-Toe Game Engine")
    print("=" * 40)
    
    # Test moves
    test_moves = [0, 4, 1, 5, 2]  # X should win with top row
    
    for move in test_moves:
        print(f"\nPlayer {game.current_player} making move at position {move}")
        success = game.make_move(move)
        if success:
            game.print_board()
        else:
            print("Invalid move!")
        
        if game.game_over:
            break
    
    print("\nGame engine test completed!")