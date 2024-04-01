import torchvision.models as models
import torch
from torchvision.models import VGG19_BN_Weights, MobileNet_V2_Weights
# def download(model_name,dummy_input):
#     # 获取PyTorch的预训练ResNet模型
#     resnet_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
#     resnet_model.eval()
#     # 转换模型
#     torch.onnx.export(resnet_model, 
#                       dummy_input, 
#                       model_name, 
#                       export_params=True, 
#                       opset_version=10, 
#                       do_constant_folding=True, 
#                       input_names = ['input'], 
#                       output_names = ['output'])
# download('mobilenet_v2.onnx',torch.randn(1, 3, 224, 224))
def download(model_name,dummy_input):
    # 获取PyTorch的预训练ResNet模型
    vgg_model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
    vgg_model.eval()
    # 转换模型
    torch.onnx.export(vgg_model, 
                      dummy_input, 
                      model_name, 
                      export_params=True, 
                      opset_version=10, 
                      do_constant_folding=True, 
                      input_names = ['input'], 
                      output_names = ['output'])
download('vgg19_bn.onnx',torch.randn(1, 3, 224, 224))

