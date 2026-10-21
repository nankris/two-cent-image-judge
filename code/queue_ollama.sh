#!/bin/bash
# Sequential queue (one Ollama model loaded at a time; box is shared).
cd "$(dirname "$0")/.."; P=/workspace/tig-env/bin/python
for arm in d1-omni d1; do $P code/run_ollama_arms.py $arm --controls; $P code/run_ollama_arms.py $arm; done
$P code/run_ollama_arms.py d1 --reversed
$P code/run_ollama_arms.py qwen3-vl-2b --controls; $P code/run_ollama_arms.py qwen3-vl-2b --subset20
echo QUEUE1_DONE
