# material
segmentation of material
1、这是我毕设形谱融合算法部分的实验代码，模型结构见 形谱融合模型各结构示意图.png 和 形谱融合模型结构.png，模型代码在spaceSpectrumFusionNet.py，训练代码为skinClothWater18PatchTwoBranch.py
2、patch类训练对比模型有DBDA SSRN FDSSC DBMA 嵌入式选择模型中的分类模块（MaterialSubModel），skinClothWater18Patch.py为训练代码；
3、patch_pred_whole_picture.py为DBDA之类的预测代码，因为其含有全连接层，需要分割为patch来一个个预测
4、skinClothWater18Test.py为整图预测代码
5、skinClothWater18Whole.py为整图训练的代码：PPLiteSEG\BiSeNetV2\SSDGL\FreeNet
6、waterAndSkinModel128.py为128波段高光谱图像的训练和响应的测试代码：waterAndSkinModel128Test.py，使用的是MaterialSubModel模型
7、utils中有各种函数的实现，如：渲染、加载光谱图像、矩形框绘制、形态学操作、标签验证、分割训练集和测试集、绘制光谱曲线、focal loss、
8、distributeTrain文件夹下为一些分布式训练代码，但是感觉没必要分布式训练，代码能跑起来，之前我也跑过。
9、data文件夹下为 训练集、测试集划分的全局列表
10、WaterCode为崑哥代码
