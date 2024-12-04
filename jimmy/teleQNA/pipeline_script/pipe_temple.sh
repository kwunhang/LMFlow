#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export DS_SKIP_CUDA_CHECK=1

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=docker0 # check by cmd ifconfig eno3:
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export WANDB_MODE=disabled


#1 
./jimmy/teleQNA/ft_script/


#2
export CUDA_VISIBLE_DEVICES=
#3
./jimmy/teleQNA/test_script/test_ .sh > /home/jimmy/LLM-FT/teleQNA_result/acc_ .txt
