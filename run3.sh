#!/bin/bash
# WorldMind Universal Run Script
# Supports all three environments: Alfred (eb-alf), Habitat (eb-hab), Navigation (eb-nav)

set -e

# ============================================================
# ENVIRONMENT VARIABLES (Export Section)
# ============================================================

# Display settings (for headless environments)
export DISPLAY=":1"

# GPU configuration
export CUDA_VISIBLE_DEVICES="0"

# API configuration (required - set these before running)
export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://api.openai.com/v1"

# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

# Model configuration
MODEL_NAME="gpt-4o-mini"

# Experiment parameters
ENV="${1}"
EXP_NAME="${2}"
ENABLE_WORLDMIND="${3}"

# Set defaults if not provided
if [ -z "$ENV" ]; then
    ENV="eb-hab"
fi

if [ -z "$EXP_NAME" ]; then
    EXP_NAME="baseline"
fi

if [ -z "$ENABLE_WORLDMIND" ]; then
    ENABLE_WORLDMIND="True"
fi

# WorldMind component models (fixed to MODEL_NAME)
export WORLDMIND_DISCRIMINATOR_MODEL="$MODEL_NAME"
export WORLDMIND_SUMMARIZER_MODEL="$MODEL_NAME"
export WORLDMIND_REFLECTOR_MODEL="$MODEL_NAME"
export WORLDMIND_REFINER_MODEL="$MODEL_NAME"

# ============================================================
# VALIDATION
# ============================================================

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "=========================================="
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "=========================================="
    echo "Please set it before running:"
    echo "  export OPENAI_API_KEY=your_api_key_here"
    echo "  export OPENAI_BASE_URL=https://api.openai.com/v1  # Optional"
    echo ""
    exit 1
fi

# Validate environment
case "$ENV" in
    eb-alf|eb-hab|eb-nav)
        echo "✓ Valid environment: $ENV"
        ;;
    *)
        echo "=========================================="
        echo "ERROR: Invalid environment '$ENV'"
        echo "=========================================="
        echo "Valid options: eb-alf, eb-hab, eb-nav"
        echo "Usage: $0 [ENV] [EXP_NAME] [ENABLE_WORLDMIND]"
        echo "Example: $0 eb-hab my_experiment True"
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
