#!/bin/bash

export DISPLAY=:1

# 统一设置模型名称   DeepSeek-V3.2-Exp  
MODEL_NAME="DeepSeek-V3.2-Exp"

export OPENAI_API_KEY=VVcFwGbc8Y6iV68ZN4V1Zf2nFAGEK6qD 
export OPENAI_BASE_URL=https://antchat.alipay.com/v1

export AWM_DISCRIMINATOR_MODEL=$MODEL_NAME
export AWM_SUMMARIZER_MODEL=$MODEL_NAME
export AWM_REFLECTOR_MODEL=$MODEL_NAME
export AWM_REFINER_MODEL=$MODEL_NAME

export CUDA_VISIBLE_DEVICES=1

python -m embodiedbench.main \
    env=eb-hab \
    enable_awm=True \
    model_name=$MODEL_NAME \
    exp_name='multi_awm2' \
    resume=True 
