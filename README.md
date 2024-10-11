# 智能问答机器人


## 运行

推荐使用pmd

```shell
pip install pdm
```

安装依赖

```shell
pdm install
pdm run ./src/data.py
pdm run ./src/robot.py
```


保证vscode的`Pylance`可以正常找到`pdm`安装的依赖。

```
pdm venv create
pdm sync    
```