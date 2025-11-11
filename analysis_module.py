# analysis_module.py
"""
Enhanced analysis and visualization module for Tic-Tac-Toe AI comparison.
Generates Matplotlib charts and diagrams for the final report with experimental data.
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

    def create_performance_comparison_chart(self):
        """Create enhanced bar charts comparing algorithm performance based on experimental data."""
        # actual experimental data
        algorithms = ['BFS', 'Greedy', 'Minimax']
    
        # experimental averages
        avg_nodes = [765, 19, 4589]  # From game data
        avg_times = [0.005, 0.001, 0.022]  # From game data
        
        # Win/Draw/Loss rates 
        win_rates = [67, 67, 67]      # All have same win rate
        draw_rates = [0, 33, 33]      # Draw rates differ
        loss_rates = [33, 0, 0]       # Loss rates reveal true differences
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))  
        
        # Colors matching Pygame interface
        colors = ['#64FF64', '#FFA500', '#B464F0']  
        outcome_colors = ['#2E8B57', '#FFD700', '#FF4444']  # Green (win), Gold (draw), Red (loss)
        
        # Nodes evaluated comparison (keep same)
        bars1 = ax1.bar(algorithms, avg_nodes, color=colors, alpha=0.8)
        ax1.set_title('Average Nodes Evaluated per Game', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Nodes', fontsize=12, labelpad=15) 
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(avg_nodes)*0.01,
                    f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Computation time comparison 
        bars2 = ax2.bar(algorithms, avg_times, color=colors, alpha=0.8)
        ax2.set_title('Average Computation Time per Game', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12, labelpad=15) 
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_times)*0.01,
                    f'{height:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        # Outcome Distribution (Stacked Bar Chart)
        bar_width = 0.6
        x_pos = np.arange(len(algorithms))
        
        # Create stacked bars
        bars_win = ax3.bar(x_pos, win_rates, bar_width, label='Wins', color=outcome_colors[0], alpha=0.8)
        bars_draw = ax3.bar(x_pos, draw_rates, bar_width, bottom=win_rates, label='Draws', color=outcome_colors[1], alpha=0.8)
        bars_loss = ax3.bar(x_pos, loss_rates, bar_width, bottom=np.array(win_rates) + np.array(draw_rates), 
                        label='Losses', color=outcome_colors[2], alpha=0.8)
        
        ax3.set_title('Game Outcome Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Percentage (%)', fontsize=12, labelpad=15) 
        ax3.set_ylim(0, 100)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(algorithms)
        ax3.grid(axis='y', alpha=0.3)
        ax3.legend(loc='upper right')
        
        # Add value annotations on each segment
        for i, (win, draw, loss) in enumerate(zip(win_rates, draw_rates, loss_rates)):
            ax3.text(i, win/2, f'{win}%', ha='center', va='center', fontweight='bold', color='white')
            ax3.text(i, win + draw/2, f'{draw}%', ha='center', va='center', fontweight='bold', color='black')
            if loss > 0:
                ax3.text(i, win + draw + loss/2, f'{loss}%', ha='center', va='center', fontweight='bold', color='white')
        
        plt.tight_layout(pad=3.0) 
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def create_search_efficiency_plot(self):
        """Create line plot showing search efficiency across different game phases."""
        # Define test positions representing different game phases
        test_positions = [
            # Position 0: Empty board (start of game)
            ([' '] * 9, 'X'),
            # Position 1: Early game (one move made)
            (['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 'O'),
            # Position 2: Mid game (strategic position)
            (['X', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'X'], 'O'),
            # Position 3: Near win situation
            (['X', 'X', ' ', ' ', 'O', 'O', ' ', ' ', ' '], 'X'),
            # Position 4: Defensive position
            (['X', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'X'], 'O')
        ]
        
        # Benchmark algorithms on these positions
        results = self.benchmark_algorithms(test_positions)
        
        # Create the line plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        algorithms = ['BFS', 'Greedy', 'Minimax']
        colors = ['#64FF64', '#FFA500', '#B464F0']  # Green, Orange, Purple
        markers = ['o', 's', '^']  # Circle, Square, Triangle
        line_styles = ['-', '--', '-.']  # Solid, Dashed, Dash-dot
        
        # Game phase descriptions for x-axis labels
        phase_descriptions = [
            'Empty Board\n(Start)',
            'Early Game\n(1 move)',
            'Mid Game\n(Strategic)',
            'Near Win\n(Attack)',
            'Defensive\n(Block)'
        ]
        
        for i, algo in enumerate(algorithms):
            positions = range(len(results))
            nodes = [result[algo]['nodes_evaluated'] for result in results]
            
            # Plot the line with markers
            ax.plot(positions, nodes, 
                   marker=markers[i], 
                   color=colors[i], 
                   linestyle=line_styles[i],
                   label=algo, 
                   linewidth=3, 
                   markersize=10,
                   markeredgecolor='black',
                   markeredgewidth=1)
        
        # Customize the plot
        ax.set_xlabel('Game Phase and Position Complexity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nodes Evaluated', fontsize=12, fontweight='bold')
        ax.set_title('Search Efficiency Across Different Game Phases', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis labels with phase descriptions
        ax.set_xticks(range(len(phase_descriptions)))
        ax.set_xticklabels(phase_descriptions, fontsize=10)
        
        # Use log scale for y-axis to handle Minimax's large numbers
        ax.set_yscale('log')
        ax.set_ylabel('Nodes Evaluated (Log Scale)', fontsize=12, fontweight='bold')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
        
        # Add value annotations on points
        for i, algo in enumerate(algorithms):
            positions = range(len(results))
            nodes = [result[algo]['nodes_evaluated'] for result in results]
            
            for j, (x, y) in enumerate(zip(positions, nodes)):
                # Position text slightly above the point
                va = 'bottom'
                offset = 0.1 * y  # Dynamic offset based on value
                
                ax.annotate(f'{y:,}', 
                           xy=(x, y), 
                           xytext=(0, 10 if i != 2 else -15),  # Adjust for Minimax
                           textcoords='offset points',
                           ha='center', 
                           va=va,
                           fontsize=9,
                           fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor=colors[i], 
                                   alpha=0.7,
                                   edgecolor='black'))
        
        # Add explanatory text
        plt.figtext(0.5, 0, 
                   'Key Insight: Algorithms show distinct computational patterns -\n Greedy maintains minimal evaluation, BFS demonstrates adaptive intelligence, while Minimax exhibits extreme computational demands',
                   ha='center', fontsize=10, style='italic', color='red', weight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15) 
        plt.savefig('search_efficiency_phases.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
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
    
    def create_pruning_efficiency_chart(self):
        """Create visualization showing alpha-beta pruning efficiency across different game phases."""
        
        test_positions = [
            # Early game (deep search)
            (['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 'O'),
            # Mid game (medium search)  
            (['X', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'X'], 'O'),
            # Near win (shallow search)
            (['X', 'X', ' ', ' ', 'O', 'O', ' ', ' ', ' '], 'X'),
        ]
        
        position_names = ['Early Game', 'Mid Game', 'Near Win']
        nodes_with_pruning = []
        nodes_without_pruning = []
        
        print("Testing pruning efficiency...")
        for i, (board_config, player) in enumerate(test_positions):
            # Test WITH pruning
            self.game.board = board_config[:]
            self.game.current_player = player
            move, nodes_pruning, time_pruning = self.minimax_ai.find_move()
            nodes_with_pruning.append(nodes_pruning)
            print(f"With pruning: {nodes_pruning} nodes")
            
            # Test WITHOUT pruning
            self.game.board = board_config[:]  # Reset
            self.game.current_player = player
            move, nodes_no_pruning, time_no_pruning = self.minimax_ai.find_move_no_pruning()
            nodes_without_pruning.append(nodes_no_pruning)
            print(f"Without pruning: {nodes_no_pruning} nodes")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Node comparison
        x_pos = range(len(position_names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos, nodes_without_pruning, width, label='Without Pruning', color='red', alpha=0.7)
        bars2 = ax1.bar([p + width for p in x_pos], nodes_with_pruning, width, label='With Alpha-Beta', color='green', alpha=0.7)
        
        ax1.set_xlabel('Game Phase')
        ax1.set_ylabel('Nodes Evaluated')
        ax1.set_title('Alpha-Beta Pruning: Node Reduction', fontsize=14, fontweight='bold')
        ax1.set_xticks([p + width/2 for p in x_pos])
        ax1.set_xticklabels(position_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(nodes_without_pruning)*0.01,
                        f'{height:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Pruning efficiency
        pruning_efficiency = [(1 - nodes_with_pruning[i]/nodes_without_pruning[i]) * 100 
                            for i in range(len(nodes_with_pruning))]
        
        bars3 = ax2.bar(x_pos, pruning_efficiency, color='purple', alpha=0.7)
        ax2.set_xlabel('Game Phase')
        ax2.set_ylabel('Nodes Eliminated (%)')
        ax2.set_title('Pruning Efficiency', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(position_names)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'Saved {nodes_without_pruning[i]-nodes_with_pruning[i]:,} nodes', 
                    ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('pruning_efficiency.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def create_comprehensive_summary_table(self):
        """Create a professional summary table of algorithm properties and performance."""
        # Algorithm properties and experimental results
        table_data = [
            {
                'Algorithm': 'BFS',
                'Search Type': 'Uninformed',
                'Optimal': 'No',
                'Complete': 'Yes',
                'Avg Nodes/Game': '765',
                'Avg Time/Game': '0.005s',
                'Win Rate': '67%',
                'Draw Rate': '0%',
                'Loss Rate': '33%',
                'Strength': 'Excellent mistake exploitation',
                'Weakness': 'Vulnerable to optimal play',
                'Practical Performance': 'Good(Opportunistic)'
            },
            {
                'Algorithm': 'Greedy',
                'Search Type': 'Informed',
                'Optimal': 'No',
                'Complete': 'No',
                'Avg Nodes/Game': '19',
                'Avg Time/Game': '0.001s',
                'Win Rate': '67%',
                'Draw Rate': '33%',
                'Loss Rate': '0%',
                'Strength': 'Never loses, extremely fast',
                'Weakness': 'Risk-averse, frequent draws',
                'Practical Performance': 'Good(Aggressive)'
            },
            {
                'Algorithm': 'Minimax',
                'Search Type': 'Adversarial',
                'Optimal': 'Yes',
                'Complete': 'Yes',
                'Avg Nodes/Game': '4,589',
                'Avg Time/Game': '0.022s',
                'Win Rate': '67%',
                'Draw Rate': '33%',
                'Loss Rate': '0%',
                'Strength': 'Never loses',
                'Weakness': '241x computational overhead',
                'Practical Performance': 'Excellent(Conservative)'
            }
        ]
        
        df = pd.DataFrame(table_data)
        
        # Create the table visualization 
        fig, ax = plt.subplots(figsize=(18, 6)) 
        
        ax.axis('tight')
        ax.axis('off')
    
        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.06, 0.07, 0.05, 0.05, 0.11, 0.11, 0.05, 0.05, 0.05, 0.14, 0.14, 0.14]
            ###colWidths=[0.07, 0.08, 0.1, 0.1, 0.08, 0.08, 0.06, 0.15, 0.15, 0.1] 
        )
    
        # Style the table 
        table.auto_set_font_size(False)
        table.set_fontsize(9)  
        table.scale(1, 1.8)    
    
        # Header styling 
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white', size=8)  
    
        
        # Row styling with alternating colors
        for i in range(1, len(df) + 1):
            color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
                
                # Highlight key performance metrics
                cell_text = table[(i, j)].get_text().get_text()
                if '0%' in cell_text and 'Loss' in df.columns[j]:
                    table[(i, j)].set_text_props(weight='bold', color='green')
                elif '33%' in cell_text and 'Loss' in df.columns[j]:
                    table[(i, j)].set_text_props(weight='bold', color='red')
                elif '67%' in cell_text and 'Win' in df.columns[j]:
                    table[(i, j)].set_text_props(weight='bold', color='blue')
        
        plt.title('Tic-Tac-Toe AI Algorithm Comprehensive Comparison\nTheory vs. Practical Performance', 
             fontsize=14, fontweight='bold', pad=10)
    
        # Add both key findings and conclusion
        plt.figtext(0.5, 0.3, 
                'Key Finding: All algorithms achieved 67% win rates, but loss/draw profiles reveal fundamental risk differences',
                ha='center', fontsize=9, style='italic', color='red', weight='bold')
        
        plt.figtext(0.5, 0.2, 
                'Insight: Minimax demonstrates risk-averse optimality (0% losses) while Greedy shows aggressive local optimization and BFS provides balanced efficiency',
                ha='center', fontsize=9, style='italic', color='blue', weight='bold')
        
        plt.figtext(0.5, 0.1, 
                'Conclusion: Algorithm selection depends on strategic philosophy - Minimax for safety, Greedy for aggression, BFS for balanced performance',
                ha='center', fontsize=9, style='italic', color='green', weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)  
        plt.savefig('algorithm_comparison_table.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return df
    
    def generate_experimental_report(self):
        """Generate a comprehensive report based on experimental data."""
        print("Generating visualizations...")  
    
    # Generate visualizations
        self.create_performance_comparison_chart()
        self.create_search_efficiency_plot()  
        self.create_pruning_efficiency_chart() ### testingggg november 3
        self.create_comprehensive_summary_table()

# Main execution
if __name__ == "__main__":
    print("Enhanced Tic-Tac-Toe AI Analysis Module")  
    print("=" * 50)  
    
    analyzer = TicTacToeAnalyzer()
    
    # Generate comprehensive experimental report
    analyzer.generate_experimental_report()