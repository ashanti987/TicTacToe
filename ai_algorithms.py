# ai_algorithms.py
"""
AI algorithms for Tic-Tac-Toe:
- Breadth-First Search (BFS) for shortest path to win
- Greedy Best-First Search with heuristic evaluation
- Minimax with Alpha-Beta Pruning for optimal play
"""

from game_engine import get_board_hash, get_opponent
from collections import deque
import time

class TicTacToeAI:
    def __init__(self, game_engine):
        """Initialize AI with access to game engine."""
        self.game = game_engine
        self.nodes_evaluated = 0
        self.computation_time = 0
    
    def reset_stats(self):
        """Reset evaluation statistics."""
        self.nodes_evaluated = 0
        self.computation_time = 0

class BFSAI(TicTacToeAI):
    """Breadth-First Search AI - finds shortest path to win."""
    
    def find_move(self):
        """
        Find the move that leads to shortest path to victory.
        Returns (move, nodes_evaluated, computation_time)
        """
        start_time = time.time()
        self.reset_stats()
        
        current_player = self.game.current_player
        available_moves = self.game.get_available_moves()
        
        if not available_moves:
            return None, 0, 0
        
        # For each possible move, perform BFS to find shortest path to win
        best_move = None
        shortest_path_length = float('inf')
        
        for move in available_moves:
            # Simulate making this move
            temp_board = self.game.board[:]
            temp_board[move] = current_player
            
            ##  i just added nodes searched hee
            path_length, nodes_searched = self._bfs_shortest_path(temp_board, current_player)
            self.nodes_evaluated += nodes_searched ## i just changed this from +=1 to = nodes_searched
            
            if path_length is not None and path_length < shortest_path_length:
                shortest_path_length = path_length
                best_move = move
        
        # If no winning path found, return first available move
        if best_move is None:
            best_move = available_moves[0]
        
        self.computation_time = time.time() - start_time
        return best_move, self.nodes_evaluated, self.computation_time
    
    def _bfs_shortest_path(self, board, player):
        """
        Perform BFS to find shortest path to win from given board state.
        Returns length of shortest winning path or None if no win possible.
        """
        queue = deque()
        visited = set()
        nodes_searched = 0 ## just added this here. tracks nodesin BFS

        initial_state = (tuple(board), player, 0)  # (board, current_player, path_length)
        queue.append(initial_state)
        visited.add(tuple(board))
        nodes_searched += 1 ## just added this here
        
        while queue:
            current_board, current_player, path_length = queue.popleft()
            ## just added
            nodes_searched += 1
            ##just added 

            # Check if this is a winning state for the original player
            if self._is_winning_state(current_board, player):
                return path_length, nodes_searched ##just added noodes search here
            
            # Get available moves
            available_moves = [i for i, spot in enumerate(current_board) if spot == ' ']
            
            for move in available_moves:
                # Make move
                new_board = list(current_board)
                new_board[move] = current_player
                new_board_tuple = tuple(new_board)
                
                if new_board_tuple not in visited:
                    visited.add(new_board_tuple)
                    next_player = get_opponent(current_player)
                    queue.append((new_board_tuple, next_player, path_length + 1))
        
        return None  # No winning path found
    
    def _is_winning_state(self, board, player):
        """Check if the board is a winning state for the given player."""
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns  
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] == player:
                return True
        return False

class GreedyAI(TicTacToeAI):
    """Greedy Best-First Search AI - uses heuristic to choose best immediate move."""
    
    def find_move(self):
        """
        Find move with highest immediate heuristic value.
        Returns (move, nodes_evaluated, computation_time)
        """
        start_time = time.time()
        self.reset_stats()
        
        current_player = self.game.current_player
        available_moves = self.game.get_available_moves()
        
        if not available_moves:
            return None, 0, 0
        
        best_move = None
        best_heuristic = float('-inf')
        
        for move in available_moves:
            # Simulate making this move
            temp_board = self.game.board[:]
            temp_board[move] = current_player
            
            # Evaluate heuristic for this board state
            heuristic_val = self._evaluate_heuristic(temp_board, current_player)
            self.nodes_evaluated += 1
            
            if heuristic_val > best_heuristic:
                best_heuristic = heuristic_val
                best_move = move
        
        self.computation_time = time.time() - start_time
        return best_move, self.nodes_evaluated, self.computation_time
    
    def _evaluate_heuristic(self, board, player):
        """
        Evaluate board state with heuristic.
        Higher values are better for the given player.
        """
        opponent = get_opponent(player)
        score = 0
        
        # Define winning lines
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for line in lines:
            line_values = [board[i] for i in line]
            player_count = line_values.count(player)
            opponent_count = line_values.count(opponent)
            empty_count = line_values.count(' ')
            
            # Score based on potential for winning
            if opponent_count == 0:
                # No opponent pieces - this line is promising
                score += player_count ** 2
            elif player_count == 0:
                # Opponent has pieces here - defensive penalty
                score -= opponent_count ** 2
        
        # Center control bonus
        if board[4] == player:  # Center position
            score += 3
        
        return score

class MinimaxAI(TicTacToeAI):
    """Minimax with Alpha-Beta Pruning AI - optimal play."""
    
    def find_move(self):
        """
        Find optimal move using Minimax with Alpha-Beta pruning.
        Returns (move, nodes_evaluated, computation_time)
        """
        start_time = time.time()
        self.reset_stats()
        
        current_player = self.game.current_player
        available_moves = self.game.get_available_moves()
        
        if not available_moves:
            return None, 0, 0
        
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in available_moves:
            # Simulate making this move
            temp_board = self.game.board[:]
            temp_board[move] = current_player
            
            # Evaluate this move with minimax
            move_value = self._minimax_alpha_beta(
                temp_board, 
                get_opponent(current_player), 
                True,  # CHANGED to True because AI (O) is maximizing player
                alpha, 
                beta,
                depth=0
            )
            self.nodes_evaluated += 1
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
                alpha = max(alpha, best_value)
        
        self.computation_time = time.time() - start_time
        return best_move, self.nodes_evaluated, self.computation_time
    
    def _minimax_alpha_beta(self, board, current_player, is_maximizing, alpha, beta, depth):
        """Minimax algorithm with Alpha-Beta pruning."""
        self.nodes_evaluated += 1
        
        # Check terminal states
        game_result = self._check_terminal_state(board)
        if game_result is not None:
            return game_result
        
        available_moves = [i for i, spot in enumerate(board) if spot == ' ']
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in available_moves:
                new_board = list(board)
                new_board[move] = current_player
                
                eval = self._minimax_alpha_beta(
                    new_board, 
                    get_opponent(current_player), 
                    False, 
                    alpha, 
                    beta,
                    depth + 1
                )
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in available_moves:
                new_board = list(board)
                new_board[move] = current_player
                
                eval = self._minimax_alpha_beta(
                    new_board, 
                    get_opponent(current_player), 
                    True, 
                    alpha, 
                    beta,
                    depth + 1
                )
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def _check_terminal_state(self, board):
        """Check if board is terminal state and return utility value."""
        # Check wins
        for player in ['X', 'O']:
            if self._is_winning_state(board, player):
                return 10 if player == 'O' else -10  # Assuming O is AI
        
        # Check draw
        if ' ' not in board:
            return 0
        
        return None  # Game continues
    
    def _is_winning_state(self, board, player):
        """Check if the board is a winning state for the given player."""
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns  
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] == player:
                return True
        return False

# Test the AI algorithms
if __name__ == "__main__":
    from game_engine import TicTacToe
    
    print("Testing AI Algorithms")
    print("=" * 50)
    
    # Create a test game
    game = TicTacToe()
    
    # Test each AI
    bfs_ai = BFSAI(game)
    greedy_ai = GreedyAI(game)
    minimax_ai = MinimaxAI(game)
    
    # Test position: empty board
    print("Empty board test:")
    move, nodes, time_taken = bfs_ai.find_move()
    print(f"BFS AI: Move={move}, Nodes={nodes}, Time={time_taken:.4f}s")
    
    move, nodes, time_taken = greedy_ai.find_move()
    print(f"Greedy AI: Move={move}, Nodes={nodes}, Time={time_taken:.4f}s")
    
    move, nodes, time_taken = minimax_ai.find_move()
    print(f"Minimax AI: Move={move}, Nodes={nodes}, Time={time_taken:.4f}s")
    
    # Test position: mid-game scenario
    print("\nMid-game test:")
    game.board = ['X', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'X']
    game.current_player = 'O'
    game.game_over = False
    game.winner = None
    
    move, nodes, time_taken = bfs_ai.find_move()
    print(f"BFS AI: Move={move}, Nodes={nodes}, Time={time_taken:.4f}s")
    
    move, nodes, time_taken = greedy_ai.find_move()
    print(f"Greedy AI: Move={move}, Nodes={nodes}, Time={time_taken:.4f}s")
    
    move, nodes, time_taken = minimax_ai.find_move()
    print(f"Minimax AI: Move={move}, Nodes={nodes}, Time={time_taken:.4f}s")
    
    print("\nAI algorithms test completed!")