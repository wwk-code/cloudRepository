# current command 
snpe-onnx-to-dlc --input_network model.onnx --out_node output --input_dtype -d input_ids 1,64 -d token_type_ids 1,64 -d attention_mask 1,64 -o model.dlc

# offiline model preparation
# snpe-dlc-graph-prepare --input_dlc model.dlc --use_float_io --htp_archs v75 


# running model inference on local
snpe-net-run --container dlc/model.dlc --input_list rawFiles/raw_list.txt
