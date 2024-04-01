import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor
import numpy as np
import onnx
from tvm.contrib import graph_executor

class layer_record:
    def __init__(self):
        pass
    # 层名
    name = ""
    # 输入形状
    shape = []
    # 执行时间
    time = 0
    # std_time
    std_time = 0

# 记录每个操作的性能
perf_list = []
ops = []
op_input_shape = []
tvm.autotvm.GLOBAL_SCOPE.silent = False

# 定义fvisit函数
def deal_node(node):
    if(isinstance(node, tvm.relay.expr.Call)):
        shape = []
        print(type(node), ", name: ", node.op.name)
        print("  Inputs:")
        i = 1
        for arg in node.args:
            # 打印输入的类型
            print("  - Type of input arg:", type(arg))
            if isinstance(arg, tvm.relay.expr.Var):
                # 对于变量节点，打印其名称和类型
                shape.append([int(dim) for dim in arg.type_annotation.concrete_shape])
                print(f"  OP{i} Var Name:", arg.name_hint, "Var Shape:", arg.type_annotation.concrete_shape)
                i = i + 1
            elif isinstance(arg, tvm.relay.expr.Constant):
                # 对于常量节点，保存常量的维度
                shape.append([int(dim) for dim in arg.data.shape])
                print(f"  OP{i} Constant Shape:", arg.data.shape)
                i = i + 1
        ops.append(node)
        print(node.op)
        op_input_shape.append(shape)

def compile_and_run_each_layer(path, input_shape, target):
    onnx_model = onnx.load(path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape={'input': input_shape})
    # 获取每个操作, node类型包括：tvm.relay.expr.Var，tvm.ir.op.Op，tvm.relay.expr.Constant，tvm.relay.expr.Call，tvm.relay.function.Function等
    relay.analysis.post_order_visit(mod['main'], deal_node)
    ctx = tvm.device(target, 0)
    for i, op in enumerate(ops):
        # 跳过没有属性的操作（如tuple等）
        if not hasattr(op.op, 'name'):
            continue
        # 创建一个子图，包含从输入到当前操作的所有节点
        # print(relay.analysis.free_vars(op))
        subgraph = relay.Function(relay.analysis.free_vars(op), op)
        # print(f'subgraph{i}: ', subgraph)
        # 编译子图
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(subgraph, target)
        subgraph_module = graph_executor.GraphModule(lib['default'](ctx))
        subgraph_module.run() # 预热
        # 评估性能
        ftimer = subgraph_module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # milliseconds
        print(f"From Layer 0 To Layer {i} ({op.op.name}) inference time (mean): {np.mean(prof_res):.2f} ms, inference time (std): {np.std(prof_res):.2f} ms")
        lay = layer_record()
        lay.name = "Layer" + str(i) + op.op.name
        lay.shape = mod['main']
        lay.time = np.mean(prof_res)
        lay.std_time = np.std(prof_res)
        perf_list.append(lay)

input_shape = (1, 3, 224, 224)
# CPU运行
# compile_and_run_each_layer('resnet50.onnx', input_shape, 'llvm')
# cpu = perf_list
# # 将perf_list的内容打印出来并保存到csv
# for i in cpu:
#     print("GPU Layer Name: ", i.name, "Time: ", i.time, "Std Time: ", i.std_time)
# # 将perf_list的内容保存到csv
import csv
# with open('perf_cpu.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Layer Name", "Time_CPU", "Std_Time_CPU"])
#     for i in cpu:
#         writer.writerow([i.name, i.time, i.std_time])
perf_list = []
ops = []

# GPU运行
compile_and_run_each_layer('resnet50.onnx', input_shape, 'cuda')
gpu = perf_list
# 将perf_list的内容打印出来
for i in gpu:
    print("GPU Layer Name: ", i.name, "Time: ", i.time, "Std Time: ", i.std_time)

# 将perf_list的内容保存到csv 
with open('perf_gpu.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Layer Name", "Time_GPU", "Std_Time_GPU"])
    for i in gpu:
        writer.writerow([i.name, i.time, i.std_time])
perf_list = []