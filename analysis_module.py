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
        # Use your actual experimental data
        algorithms = ['BFS', 'Greedy', 'Minimax']
        
        # Your experimental averages
        avg_nodes = [19, 19, 2500]  # From your data
        avg_times = [0.0064, 0.0003, 0.0133]  # From your data
        win_rates = [67, 33, 0]  # From your data
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Colors matching your Pygame interface
        colors = ['#64FF64', '#FFA500', '#B464F0']  # Green, Orange, Purple
        
        # Nodes evaluated comparison
        bars1 = ax1.bar(algorithms, avg_nodes, color=colors, alpha=0.8)
        ax1.set_title('Average Nodes Evaluated per Game', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Nodes', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(avg_nodes)*0.01,
                    f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Computation time comparison
        bars2 = ax2.bar(algorithms, avg_times, color=colors, alpha=0.8)
        ax2.set_title('Average Computation Time per Game', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_times)*0.01,
                    f'{height:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        # Win rate comparison
        bars3 = ax3.bar(algorithms, win_rates, color=colors, alpha=0.8)
        ax3.set_title('Win Rate Against Human Players', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Win Rate (%)', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height}%', ha='center', va='bottom', fontweight='bold',
                    color='red' if height == 0 else 'black')
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_comprehensive_summary_table(self):
        """Create a professional summary table of algorithm properties and performance."""
        # Algorithm properties and experimental results
        table_data = [
            {
                'Algorithm': 'BFS',
                'Search Type': 'Uninformed',
                'Theoretical Optimal': 'No',
                'Theoretical Complete': 'Yes',
                'Avg Nodes/Game': '19',
                'Avg Time/Game': '0.0064s',
                'Win Rate': '67%',
                'Strength': 'Excellent mistake exploitation',
                'Weakness': 'Vulnerable to optimal play',
                'Practical Performance': 'Excellent'
            },
            {
                'Algorithm': 'Greedy',
                'Search Type': 'Informed',
                'Theoretical Optimal': 'No',
                'Theoretical Complete': 'No',
                'Avg Nodes/Game': '19',
                'Avg Time/Game': '0.0003s',
                'Win Rate': '33%',
                'Strength': 'Never loses, extremely fast',
                'Weakness': 'Risk-averse, frequent draws',
                'Practical Performance': 'Good'
            },
            {
                'Algorithm': 'Minimax',
                'Search Type': 'Adversarial',
                'Theoretical Optimal': 'Yes',
                'Theoretical Complete': 'Yes',
                'Avg Nodes/Game': '2,500',
                'Avg Time/Game': '0.0133s',
                'Win Rate': '0%',
                'Strength': 'Theoretically optimal',
                'Weakness': 'Practical implementation gaps',
                'Practical Performance': 'Poor'
            }
        ]
        
        df = pd.DataFrame(table_data)
        
        # Create the table visualization - MAKE TALLER
        fig, ax = plt.subplots(figsize=(16, 6))  # Changed from (16, 8) to (16, 10)
        
        ax.axis('tight')
        ax.axis('off')
    
        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.07, 0.08, 0.1, 0.1, 0.08, 0.08, 0.06, 0.15, 0.15, 0.1] 
        )
    
        # Style the table - INCREASE FONT SIZES
        table.auto_set_font_size(False)
        table.set_fontsize(9)  # Increased from 9 to 10
        table.scale(1, 1.8)    # Reduced vertical scaling from 2 to 1.8
    
        # Header styling - INCREASE HEADER FONT SIZE
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white', size=8)  # Increased from 10 to 12
    
        
        # Row styling with alternating colors
        for i in range(1, len(df) + 1):
            color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
                
                # Highlight key findings
                cell_text = table[(i, j)].get_text().get_text()
                if '0%' in cell_text:
                    table[(i, j)].set_text_props(weight='bold', color='red')
                elif '67%' in cell_text:
                    table[(i, j)].set_text_props(weight='bold', color='green')
                elif '33%' in cell_text:
                    table[(i, j)].set_text_props(weight='bold', color='blue')
                '''elif '2,500' in cell_text:
                    table[(i, j)].set_text_props(weight='bold', color='orange')'''
        
        plt.title('Tic-Tac-Toe AI Algorithm Comprehensive Comparison\nTheory vs. Practical Performance', 
             fontsize=14, fontweight='bold', pad=10)
    
        # Add both key findings and conclusion
        plt.figtext(0.5, 0.2, 
                'Key Finding: Minimax shows 0% win rate despite theoretical optimality - revealing significant theory-practice gap',
                ha='center', fontsize=9, style='italic', color='red', weight='bold')
        
        plt.figtext(0.5, 0.1, 
                'Conclusion: Simpler algorithms (BFS, Greedy) outperformed theoretically optimal Minimax in practical gameplay',
                ha='center', fontsize=9, style='italic', color='blue', weight='bold')
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison_table.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return df
        
    def create_minimax_paradox_chart(self):
        """Create a visualization highlighting the Minimax paradox."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        algorithms = ['BFS', 'Greedy', 'Minimax']
        
        # Computational cost vs effectiveness
        computational_cost = [19, 19, 2500]  # Nodes
        effectiveness = [67, 33, 0]  # Win rate
        
        # Scatter plot: Cost vs Effectiveness
        scatter = ax1.scatter(computational_cost, effectiveness, 
                            s=300, c=['green', 'orange', 'purple'], alpha=0.7)
        ax1.set_xlabel('Computational Cost (Nodes Evaluated)', fontsize=12)
        ax1.set_ylabel('Effectiveness (Win Rate %)', fontsize=12)
        ax1.set_title('The Minimax Paradox: Cost vs Effectiveness', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')  # Log scale to handle large differences
        
        # Annotate points
        for i, algo in enumerate(algorithms):
            ax1.annotate(algo, (computational_cost[i], effectiveness[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontweight='bold', fontsize=11)
        
        # Performance across play styles
        play_styles = ['Optimal', 'Random', 'Suboptimal']
        bfs_performance = [0, 100, 100]  # Win rates
        greedy_performance = [0, 100, 0]   # Win/Draw rates
        minimax_performance = [0, 0, 0]    # Always lost
        
        x = np.arange(len(play_styles))
        width = 0.25
        
        bars1 = ax2.bar(x - width, bfs_performance, width, label='BFS', color='green', alpha=0.8)
        bars2 = ax2.bar(x, greedy_performance, width, label='Greedy', color='orange', alpha=0.8)
        bars3 = ax2.bar(x + width, minimax_performance, width, label='Minimax', color='purple', alpha=0.8)
        
        ax2.set_xlabel('Human Play Style', fontsize=12)
        ax2.set_ylabel('AI Win Rate (%)', fontsize=12)
        ax2.set_title('Algorithm Performance Across Play Styles', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(play_styles)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 120)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 3,
                            f'{height}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('minimax_paradox.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def generate_experimental_report(self):
        """Generate a comprehensive report based on experimental data."""
        print("Generating visualizations...")  
    
    # Generate visualizations
        self.create_performance_comparison_chart()
        self.create_minimax_paradox_chart()
        self.create_comprehensive_summary_table()

# Main execution
if __name__ == "__main__":
    print("Enhanced Tic-Tac-Toe AI Analysis Module")  # KEEP
    print("=" * 50)  # KEEP
    
    analyzer = TicTacToeAnalyzer()
    
    # Generate comprehensive experimental report
    analyzer.generate_experimental_report()