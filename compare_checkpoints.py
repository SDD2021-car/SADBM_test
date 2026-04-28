import torch

# 加载两个模型的权重
model_1 = torch.load('/data/yjy_data/DDBM_GT_Unet/logs_SEN12_0214_GT_se_SAB/model_2_010000.pt')
model_2 = torch.load('/data/yjy_data/DDBM_GT_Unet/logs_SEN12_0214_GT_se_SAB/model_2_030000.pt')

# 比较两个模型的权重
def compare_models(model_1, model_2):
    # 比较每一层的权重
    for key in model_1.keys():
        if key not in model_2:
            print(f"Key {key} not found in both models.")
            return False
        if torch.equal(model_1[key], model_2[key]) == False:
            print(f"Difference found in layer: {key}")
            return False
    return True

# 执行比较
if compare_models(model_1, model_2):
    print("The models are identical.")
else:
    print("The models are different.")
