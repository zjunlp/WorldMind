#!/bin/bash

export DISPLAY=:1

# 统一设置模型名称 gpt-4o-mini  gpt-4.1-mini-2025-04-14
MODEL_NAME="gpt-3.5-turbo"

export OPENAI_API_KEY=sk-vNOf3GfXPvYwpXgE4G7RHcMM1uSVr0iL5ex0zxshtekQhZQP
export OPENAI_BASE_URL=https://vip.dmxapi.com/v1

export AWM_DISCRIMINATOR_MODEL=$MODEL_NAME
export AWM_SUMMARIZER_MODEL=$MODEL_NAME
export AWM_REFLECTOR_MODEL=$MODEL_NAME
export AWM_REFINER_MODEL=$MODEL_NAME

export CUDA_VISIBLE_DEVICES=1

python -m embodiedbench.main \
    env=eb-alf \
    enable_awm=True \
    model_name=$MODEL_NAME \
    exp_name='worldmind4-1' \
    enable_multi_simulation=False 
    

 
