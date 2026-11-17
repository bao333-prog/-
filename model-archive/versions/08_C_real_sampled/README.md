# 十四类食材识别实拍适配

安装Python依赖后在此目录执行 pip install -e .。
预测命令为 python predict.py --model models/model_C_real.pt --image 实拍图片.jpg。
输出十四类之一及已知类别相对分数，仅识别食材。南瓜和整鸡仍保留在类别中，本批缺其实际图片。
实拍诊断包含训练图和同源图，实际泛化需要新的独立拍摄验证。
详细结果见records/实拍适配结果.md。配置和清单含开发机原始路径，复训前需准备数据和修改路径副本。
