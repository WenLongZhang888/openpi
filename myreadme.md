## Git管理

拉取原代码
``` shell
git checkout main
git pull upstream main

git checkout myself # 切换为自己开发分支
```

## 项目逻辑

### 配置文件
src/openpi/training/config.py 中包含了多个训练实例化配置，每个实例指定了模型架构（model）、数据管道（data）、优化器设置（lr_schedule和optimizer）以及weight_loader(从检查点加载权重)等等。

### 模型
所有模型继承于一个BaseModel抽象类（src/openpi/models/model.py(1-100)）

