#!/bin/bash

# Script to run all plotting scripts in the final_paper_plots directory

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running all plotting scripts..."
echo ""

# Run each plotting script
python "$SCRIPT_DIR/plot_classification_eval_all_models.py"
echo "✓ plot_classification_eval_all_models.py completed"

python "$SCRIPT_DIR/plot_personaqa_results_all_models.py"
echo "✓ plot_personaqa_results_all_models.py completed"

python "$SCRIPT_DIR/plot_secret_keeping_results.py"
echo "✓ plot_secret_keeping_results.py completed"

python "$SCRIPT_DIR/plot_qwen3-8b_eval_results.py"
echo "✓ plot_qwen3-8b_eval_results.py completed"

python "$SCRIPT_DIR/plot_personaqa_knowledge_eval_all_models.py"
echo "✓ plot_personaqa_knowledge_eval_all_models.py completed"

python "$SCRIPT_DIR/plot_model_progression_line_chart_shapes.py"
echo "✓ plot_model_progression_line_chart_shapes.py completed"

python "$SCRIPT_DIR/plot_all_data_diversity.py"
echo "✓ plot_all_data_diversity.py completed"

python "$SCRIPT_DIR/plot_combined_sequence_vs_token.py"
echo "✓ plot_combined_sequence_vs_token.py completed"

echo "Done!"
