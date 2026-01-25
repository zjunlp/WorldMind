#!/bin/bash
# WorldMind Universal Run Script
# Supports all three environments: Alfred (eb-alf), Habitat (eb-hab), Navigation (eb-nav)

set -e

# ============================================================
# ENVIRONMENT VARIABLES (Export Section)
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="your-openai-base-url"

# ============================================================
# CONFIGURATION PARAMETERS (Edit here)
# ============================================================

MODEL_NAME="gpt-3.5-turbo"   # Choose your model
ENV="eb-hab"              # Options: eb-alf, eb-hab, eb-nav
EXP_NAME="test"       # Your experiment name
ENABLE_WORLDMIND="True"   # True or False

# WorldMind component models (fixed to MODEL_NAME)
export WORLDMIND_DISCRIMINATOR_MODEL="$MODEL_NAME"
export WORLDMIND_SUMMARIZER_MODEL="$MODEL_NAME"
export WORLDMIND_REFLECTOR_MODEL="$MODEL_NAME"
export WORLDMIND_REFINER_MODEL="$MODEL_NAME"

# ============================================================
# VALIDATION
# ============================================================

if [ -z "$OPENAI_API_KEY" ]; then
    echo "=========================================="
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "=========================================="
    exit 1
fi

case "$ENV" in
    eb-alf|eb-hab|eb-nav)
        echo "✓ Valid environment: $ENV"
        ;;
    *)
        echo "=========================================="
        echo "ERROR: Invalid environment '$ENV'"
        echo "=========================================="
        echo "Valid options: eb-alf, eb-hab, eb-nav"
        exit 1
        ;;
esac

# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

echo ""
echo "=========================================="
echo "WorldMind Experiment Configuration"
echo "=========================================="
echo "Environment:     $ENV"
echo "Model:           $MODEL_NAME"
echo "Experiment:      $EXP_NAME"
echo "WorldMind:       $ENABLE_WORLDMIND"
echo "----------------------------------------"
echo "GPU Device:      $CUDA_VISIBLE_DEVICES"
echo "Display:         $DISPLAY"
echo "API Base URL:    $OPENAI_BASE_URL"
echo "=========================================="
echo ""

# ============================================================
# RUN EXPERIMENT
# ============================================================

python -m embodiedbench.main \
    env="$ENV" \
    model_name="$MODEL_NAME" \
    exp_name="$EXP_NAME" \
    enable_worldmind="$ENABLE_WORLDMIND"
