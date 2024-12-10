import onnx 
model = onnx.load("model.onnx")
graph_str = onnx.helper.printable_graph(model.graph)


with open("model_structure.txt","w") as f:
    f.write(graph_str)

print('Done!')

