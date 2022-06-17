#!/bin/bash
ml CUDA
cd /scratch/project/dd-21-23/test/deepspeed-test/
deepspeed --num_gpus=4 ./deep_speed.py --deepspeed ./ds_config_gpt_j.json
deepspeed --num_gpus=4 ./deep_speed_inference.py --deepspeed ./ds_config_gpt_j.json