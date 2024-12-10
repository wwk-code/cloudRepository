# 模型量化性能记录:

| datasets                      | quantize scheme                  | dlc-quantize command precision parameters            | accuracy |
| ----------------------------- | -------------------------------- | ---------------------------------------------------- | -------- |
| validation (sample_size=1000) | ptq + qat(epoch=20)              | --weights_bitwidth8--bias_bitwidth32--act_bitwidth16 | 0.69     |
| validation (sample_size=1000) | qat-with rangeLearning(epoch=10) | --weights_bitwidth8--bias_bitwidth32--act_bitwidth16 | 0.74     |



top-sim-models

validation (sample_size=1000)     qat-with rangeLearning(epoch=10)  --weights_bitwidth8--bias_bitwidth32--act_bitwidth16    0.75
