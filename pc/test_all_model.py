import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor
import numpy as np
import onnx
from tvm.contrib import graph_executor

# 记录每个操作的性能

def compile_and_run_each_layer(path, input_shape, target):
    onnx_model = onnx.load(path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape={'input': input_shape})
    # 获取每个操作, node类型包括：tvm.relay.expr.Var，tvm.ir.op.Op，tvm.relay.expr.Constant，tvm.relay.expr.Call，tvm.relay.function.Function等
    ctx = tvm.device(target, 0)
    # 编译模型
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    graph_module = graph_executor.GraphModule(lib['default'](ctx))
    graph_module.run() # 预热
    # 评估性能
    ftimer = graph_module.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # milliseconds
    print(f"All Model on {target} inference time (mean): {np.mean(prof_res):.2f} ms, inference time (std): {np.std(prof_res):.2f} ms")

input_shape = (1, 3, 224, 224)
# CPU运行
compile_and_run_each_layer('resnet50.onnx', input_shape, 'cuda')

# GPU运行
compile_and_run_each_layer('resnet50.onnx', input_shape, 'llvm')

# 运行结果
# All Model on cuda inference time (mean): 3.72 ms, inference time (std): 0.27 ms
# All Model on llvm inference time (mean): 117.02 ms, inference time (std): 4.97 ms
