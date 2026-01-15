#!/bin/bash

export DISPLAY=:1

# 统一设置模型名称 gpt-4o-mini
MODEL_NAME="gpt-3.5-turbo"

export OPENAI_API_KEY=sk-ER4wzbooL5S5rMJ4k1TwGzuYdrk5Z5csr9nfkPgMErh6u2yo
export OPENAI_BASE_URL=https://vip.dmxapi.com/v1

export WorldMind_DISCRIMINATOR_MODEL=$MODEL_NAME
export WorldMind_SUMMARIZER_MODEL=$MODEL_NAME
export WorldMind_REFLECTOR_MODEL=$MODEL_NAME
export WorldMind_REFINER_MODEL=$MODEL_NAME

export CUDA_VISIBLE_DEVICES=1

python -m embodiedbench.main \
    env=eb-hab \
    enable_awm=True \
    model_name=$MODEL_NAME \
    exp_name='ablation-with-process1' \
    enable_multi_simulation=False 
    
