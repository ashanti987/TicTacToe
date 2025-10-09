# analysis_module.py
"""
Analysis and visualization module for Tic-Tac-Toe AI comparison.
Generates Matplotlib charts and diagrams for the final report.
"""

import matplotlib.pyplot as plt
import numpy as np
from game_engine import TicTacToe
from ai_algorithms import BFSAI, GreedyAI, MinimaxAI
import time
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.gridspec as gridspec

class TicTacToeAnalyzer:
    def __init__(self):
        self.game = TicTacToe()
        self.bfs_ai = BFSAI(self.game)
        self.greedy_ai = GreedyAI(self.game)
        self.minimax_ai = MinimaxAI(self.game)
        self.results = []
    
    def benchmark_algorithms(self, test_positions):
        """
        Benchmark all algorithms on a set of test positions.
        
        Args:
            test_positions: List of (board_config, current_player) tuples
        """
        self.results = []
        
        for i, (board_config, player) in enumerate(test_positions):
            position_results = {'position_id': i, 'board': board_config[:]}
            
            for ai_name, ai in [('BFS', self.bfs_ai), 
                              ('Greedy', self.greedy_ai), 
                              ('Minimax', self.minimax_ai)]:
                
                # Set up the test position
                self.game.board = board_config[:]
                self.game.current_player = player
                self.game.game_over = False
                self.game.winner = None
                
                # Run the algorithm
                start_time = time.time()
                move, nodes_evaluated, comp_time = ai.find_move()
                actual_time = time.time() - start_time
                
                position_results[ai_name] = {
                    'move': move,
                    'nodes_evaluated': nodes_evaluated,
                    'computation_time': comp_time,
                    'actual_time': actual_time
                }
            
            self.results.append(position_results)
        
        return self.results
    
    def create_performance_comparison_chart(self):
        """Create a bar chart comparing algorithm performance."""
        if not self.results:
            print("No results to analyze. Run benchmark_algorithms() first.")
            return
        
        # Prepare data
        algorithms = ['BFS', 'Greedy', 'Minimax']
        avg_nodes = []
        avg_times = []
        
        for algo in algorithms:
            nodes = [result[algo]['nodes_evaluated'] for result in self.results]
            times = [result[algo]['computation_time'] for result in self.results]
            avg_nodes.append(np.mean(nodes))
            avg_times.append(np.mean(times))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Nodes evaluated comparison
        bars1 = ax1.bar(algorithms, avg_nodes, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('Average Nodes Evaluated per Move')
        ax1.set_ylabel('Number of Nodes')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(avg_nodes)*0.01,
                    f'{height:,.0f}', ha='center', va='bottom')
        
        # Computation time comparison
        bars2 = ax2.bar(algorithms, avg_times, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('Average Computation Time per Move')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_times)*0.01,
                    f'{height:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_search_efficiency_plot(self):
        """Create a line plot showing search efficiency over different positions."""
        if not self.results:
            print("No results to analyze. Run benchmark_algorithms() first.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = ['BFS', 'Greedy', 'Minimax']
        colors = ['blue', 'green', 'red']
        markers = ['o', 's', '^']
        
        for i, algo in enumerate(algorithms):
            positions = range(len(self.results))
            nodes = [result[algo]['nodes_evaluated'] for result in self.results]
            
            ax.plot(positions, nodes, marker=markers[i], color=colors[i], 
                   label=algo, linewidth=2, markersize=8)
        
        ax.set_xlabel('Test Position Number')
        ax.set_ylabel('Nodes Evaluated')
        ax.set_title('Search Efficiency Across Different Game Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Use log scale due to large differences
        
        plt.tight_layout()
        plt.savefig('search_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_algorithm_strategy_analysis(self):
        """Analyze and visualize the strategic differences between algorithms."""
        # Test a specific interesting position
        test_position = ['X', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'X']
        current_player = 'O'
        
        self.game.board = test_position[:]
        self.game.current_player = current_player
        self.game.game_over = False
        self.game.winner = None
        
        moves = {}
        for ai_name, ai in [('BFS', self.bfs_ai), 
                          ('Greedy', self.greedy_ai), 
                          ('Minimax', self.minimax_ai)]:
            
            move, nodes, time_taken = ai.find_move()
            moves[ai_name] = {
                'move': move,
                'nodes': nodes,
                'time': time_taken
            }
        
        # Create strategy visualization
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[3, 1, 1, 1])
        
        # Main board with moves highlighted
        ax_board = plt.subplot(gs[0])
        self._draw_tic_tac_toe_board(ax_board, test_position, moves)
        
        # Metrics for each algorithm
        for i, (algo, data) in enumerate(moves.items()):
            ax_metric = plt.subplot(gs[i+1])
            self._draw_algorithm_metrics(ax_metric, algo, data)
        
        plt.tight_layout()
        plt.savefig('strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _draw_tic_tac_toe_board(self, ax, board, moves):
        """Draw a Tic-Tac-Toe board with algorithm moves highlighted."""
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # So (0,0) is top-left
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Algorithm Move Recommendations', fontsize=14, fontweight='bold')
        
        # Draw grid
        for i in range(1, 3):
            ax.axvline(i, color='black', linewidth=2)
            ax.axhline(i, color='black', linewidth=2)
        
        # Draw X's and O's
        for i in range(9):
            row = i // 3
            col = i % 3
            x = col + 0.5
            y = row + 0.5
            
            if board[i] == 'X':
                ax.text(x, y, 'X', fontsize=30, ha='center', va='center', 
                       color='red', fontweight='bold')
            elif board[i] == 'O':
                ax.text(x, y, 'O', fontsize=30, ha='center', va='center',
                       color='blue', fontweight='bold')
        
        # Highlight recommended moves
        colors = ['green', 'orange', 'purple']
        for i, (algo, data) in enumerate(moves.items()):
            if data['move'] is not None:
                row = data['move'] // 3
                col = data['move'] % 3
                x = col + 0.5
                y = row + 0.5
                
                circle = Circle((x, y), 0.3, fill=False, 
                              edgecolor=colors[i], linewidth=3, linestyle='--')
                ax.add_patch(circle)
                ax.text(x, y + 0.4, algo, ha='center', va='bottom', 
                       color=colors[i], fontweight='bold', fontsize=8)
    
    def _draw_algorithm_metrics(self, ax, algorithm, data):
        """Draw metrics for a single algorithm."""
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        title = f"{algorithm}\nMove: {data['move']}"
        ax.text(0.5, 0.9, title, ha='center', va='center', 
               fontweight='bold', fontsize=12)
        
        metrics = [
            f"Nodes: {data['nodes']:,}",
            f"Time: {data['time']:.4f}s"
        ]
        
        for i, metric in enumerate(metrics):
            ax.text(0.5, 0.7 - i*0.2, metric, ha='center', va='center', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                        facecolor="lightgray"))
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report with all visualizations."""
        print("Generating Comprehensive AI Analysis Report...")
        print("=" * 50)
        
        # Define test positions
        test_positions = [
            # Empty board
            ([' '] * 9, 'X'),
            # Early game
            (['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 'O'),
            # Mid game
            (['X', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'X'], 'O'),
            # Near win for X
            (['X', 'X', ' ', ' ', 'O', 'O', ' ', ' ', ' '], 'X'),
            # Defensive position
            (['X', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'X'], 'O')
        ]
        
        # Run benchmarks
        print("Running algorithm benchmarks...")
        results = self.benchmark_algorithms(test_positions)
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print("-" * 30)
        
        df_data = []
        for result in results:
            for algo in ['BFS', 'Greedy', 'Minimax']:
                df_data.append({
                    'position': result['position_id'],
                    'algorithm': algo,
                    'nodes': result[algo]['nodes_evaluated'],
                    'time': result[algo]['computation_time']
                })
        
        df = pd.DataFrame(df_data)
        summary = df.groupby('algorithm').agg({
            'nodes': ['mean', 'std', 'min', 'max'],
            'time': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        print(summary)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.create_performance_comparison_chart()
        self.create_search_efficiency_plot()
        self.create_algorithm_strategy_analysis()
        
        print("\nAnalysis complete! Check generated PNG files:")
        print("- performance_comparison.png")
        print("- search_efficiency.png") 
        print("- strategy_analysis.png")
        
        return df

# Example usage and testing
if __name__ == "__main__":
    print("Tic-Tac-Toe AI Analysis Module")
    print("=" * 40)
    
    analyzer = TicTacToeAnalyzer()
    
    # Generate comprehensive report
    results_df = analyzer.generate_comprehensive_report()
    
    # Additional analysis
    print("\nAdditional Insights:")
    print("-" * 20)
    
    # Find most computationally expensive position
    max_nodes = 0
    max_position = None
    max_algorithm = None
    
    for result in analyzer.results:
        for algo in ['BFS', 'Greedy', 'Minimax']:
            nodes = result[algo]['nodes_evaluated']
            if nodes > max_nodes:
                max_nodes = nodes
                max_position = result['position_id']
                max_algorithm = algo
    
    print(f"Most complex position: Position {max_position} for {max_algorithm}")
    print(f"Nodes evaluated: {max_nodes:,}")
    
    print("\nAnalysis module test completed!")