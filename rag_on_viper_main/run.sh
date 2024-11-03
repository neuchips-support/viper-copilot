#!/bin/bash

sudo chmod 777 /dev/neuchips_*

source ~/anaconda3/etc/profile.d/conda.sh
conda activate demo_sys

./neu_weight_util --weight ../../output_weight/taide_lx_7b_chat/ --n3kid 0
echo "============== test download weight to N3000 succeed =================="

prompt="Please translate following sentences to traditional chinese: Dongshan coffee is famous for its unique position, and the constant refinement of production methods. The flavor is admired by many caffeine afficionados."

sudo ./main --n3k_id 0 -m  ../../output_weight/taide_lx_7b_chat/taide_llama2 -p "$prompt" -c 4096 -n 48 -b 32 --temp 0.9 
echo ""
echo "============== test llm inference on N3000 succeed =================="

streamlit run main.py

