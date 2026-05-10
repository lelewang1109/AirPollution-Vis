# Block 生成独立实现

这个目录把项目里的 block 语义生成逻辑单独拆出来，方便只看“网格怎么切成 block，以及 block 怎么变成语义标签”。

## 文件

- `block_generator.py`：独立可运行脚本，不依赖 `gridvis_server.py`。
- `__init__.py`：允许在其他 Python 代码中 `from block_generation_standalone import generate_blocks`。

## 生成流程

1. 输入二维网格 `[y, x]` 或时序网格 `[time, y, x]`。
2. 按 `rows/cols` 或 `block_shape` 生成每个 block 的行列范围。
3. 对每个 block 提取：
   - 均值、标准差、最大值、最小值、分位数、极差
   - 平均梯度、边界强度、主导梯度方向
   - 高值/低值像元占比
   - 高值/低值连通域数量、最大连通域面积占比
   - Moran's I、缺失率、异常点比例
4. 通过规则分数生成标签：
   - `H`：高值热点型
   - `L`：低值冷点型
   - `G`：梯度过渡型
   - `B`：边界突变型
   - `D`：扩散分散型
   - `U`：均匀稳定型
   - `M`：混合复杂型
   - `N`：噪声不确定型
5. 输出 `label_matrix`、`blocks`、`top_blocks`、`label_distribution`、`features`、`scores` 和 `evidence`。

## 直接运行内置示例

```bash
python3 block_generation_standalone/block_generator.py
```

输出会包含：

- `grid`：block 矩阵大小和原始网格大小
- `label_distribution`：每类标签数量
- `label_matrix`：二维 block 标签矩阵
- `top_blocks`：显著性最高的 block

## 指定 block 数量

```bash
python3 block_generation_standalone/block_generator.py --rows 9 --cols 14
```

这种方式会把整个网格均分成 `9 x 14` 个 block，边缘会自动处理。

## 指定每个 block 的像元大小

```bash
python3 block_generation_standalone/block_generator.py --block-shape 16x16
```

这种方式按固定像元尺寸切分，例如每个 block 是 `16 x 16` 个格点。

## 读取外部数据

支持 `.npy`、`.csv`、`.json`：

```bash
python3 block_generation_standalone/block_generator.py data.npy --rows 8 --cols 12
python3 block_generation_standalone/block_generator.py grid.csv --block-shape 20x20
python3 block_generation_standalone/block_generator.py grid.json --output block_result.json
```

`.json` 可以直接是二维/三维数组，也可以是：

```json
{
  "data": [[1, 2, 3], [4, 5, 6]]
}
```

## 在代码中调用

```python
import numpy as np
from block_generation_standalone import generate_blocks

grid = np.random.random((72, 112))
result = generate_blocks(grid, rows=9, cols=14)

print(result["label_matrix"])
print(result["top_blocks"][0]["features"])
```

## 和主项目的关系

主项目里的完整 API 会读取 NetCDF、做区域筛选、生成热力图和 LLM 可视化策略。这个目录只保留 block 生成本身：

```text
grid / time stack -> block slices -> block features -> rule scores -> semantic labels
```
