#!/bin/bash

# Script to run all plotting scripts in the final_paper_plots directory

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running all plotting scripts..."
echo ""

# Run each plotting script
python "$SCRIPT_DIR/plot_classification_eval.py"
echo "✓ plot_classification_eval.py completed"

python "$SCRIPT_DIR/plot_gender_eval_results.py"
echo "✓ plot_gender_eval_results.py completed"

python "$SCRIPT_DIR/plot_personaqa_results.py"
echo "✓ plot_personaqa_results.py completed"

python "$SCRIPT_DIR/plot_secret_keeping_results.py"
echo "✓ plot_secret_keeping_results.py completed"

python "$SCRIPT_DIR/plot_ssc_results.py"
echo "✓ plot_ssc_results.py completed"

python "$SCRIPT_DIR/plot_taboo_eval_results.py"
echo "✓ plot_taboo_eval_results.py completed"

python "$SCRIPT_DIR/plot_qwen3-8b_eval_results.py"
echo "✓ plot_qwen3-8b_eval_results.py completed"

echo "Done!"
