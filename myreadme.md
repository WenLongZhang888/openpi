## Git管理

拉取原代码
``` shell
# 保存当前工作并提交 myself
git add .
git commit -m "提交信息"

git checkout main

git pull upstream main

# 推送更新后的main到fork 
git push origin main

git checkout myself

# 将新的main合并到myself
git merge main

# 推送到fork
git push origin myself
```

## 项目逻辑

### 配置文件
src/openpi/training/config.py 中包含了多个训练实例化配置，每个实例指定了模型架构（model）、数据管道（data）、优化器设置（lr_schedule和optimizer）以及weight_loader(从检查点加载权重)等等。

### 模型
所有模型继承于一个BaseModel抽象类（src/openpi/models/model.py(1-100)）



## aloha_sim

```bash
# 训练流程参考
# examples/aloha_sim/README.md
# 训练配置参考
# src/openpi/training/config.py
```