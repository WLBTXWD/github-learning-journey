#!/usr/bin/env bash
# 138 Prefill: DP2 x TP8. 2026-09-15 profiling case.
set -euo pipefail

ROLE=prefill
LOCAL_IP=80.5.17.118
ROOT=/home/wtr/pd-profile-118-x-20260916
MODEL=/mnt/weight/kimi-k2.6-w4a8
DFLASH_MODEL=/mnt/weight/kimi_k2.5_dflash

VLLM_DIR=/mnt/share/w00937898/vllm
ASCEND_DIR=/mnt/share/w00937898/vllm-ascend
cd "$ROOT"
mkdir -p "$ROOT/$ROLE"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

export NO_PROXY=127.0.0.1,localhost,"$LOCAL_IP"
export no_proxy="$NO_PROXY"
export GLOO_SOCKET_IFNAME=enp48s3u1u1 HCCL_SOCKET_IFNAME=enp48s3u1u1 TP_SOCKET_IFNAME=enp48s3u1u1 NIC_NAME=enp48s3u1u1
export MASTER_IP="$LOCAL_IP" LOCAL_IP HCCL_IF_IP="$LOCAL_IP"
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=204 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=512
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000 VLLM_RPC_TIMEOUT=3600000 VLLM_SERVER_DEV_MODE=1 VLLM_USE_MODELSCOPE=true

for DP_RANK in 0 1; do
  PROFILE_DIR="$ROOT/$ROLE/dp${DP_RANK}"
  LOG="$ROOT/$ROLE-dp${DP_RANK}.log"
  mkdir -p "$PROFILE_DIR"
  DEVICES="$(seq -s, $((DP_RANK * 8)) $((DP_RANK * 8 + 7)))"
  ASCEND_RT_VISIBLE_DEVICES="$DEVICES" SERVER_PORT="$((7100 + DP_RANK))" \
    nohup vllm serve "$MODEL" \
      --served-model-name Eco-Tech/Kimi-K2.6-w4a8 --allowed-local-media-path / \
      --host 0.0.0.0 --port "$((7100 + DP_RANK))" \
      --data-parallel-size 2 --data-parallel-rank "$DP_RANK" --data-parallel-address "$LOCAL_IP" --data-parallel-rpc-port 12321 \
      --tensor-parallel-size 8 --safetensors-load-strategy prefetch --enable-expert-parallel \
      --seed 1024 --max-model-len 68000 --max-num-batched-tokens 8192 --max-num-seqs 16 \
      --trust-remote-code --gpu-memory-utilization 0.94 --enforce-eager \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":1}" \
      --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":8},"decode":{"dp_size":4,"tp_size":4}}}' \
      --additional-config '{"enable_fused_mc2":1,"enable_mlapo":true}' \
      --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false}" \
      >"$LOG" 2>&1 &
  echo "prefill dp${DP_RANK}: pid=$!, log=$LOG"
done
