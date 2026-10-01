# task26100256

内存中的关键帧曲线剪贴库（单标量动画轨道），支持裁取、变速粘贴和反向粘贴。

## 模型

- `Track` 覆盖 `[0, T)`，由连续、互不重叠的片段组成；每段含左端、不含右端。
- 片段类型：`Hold`（恒定）、`Linear`（线性）、`Bezier`（二维三次贝塞尔，
  控制点 `(t0,v0),(x1,y1),(x2,y2),(t1,v1)`，满足 `t0<=x1<=x2<=t1`，纵向可过冲）。

## API

- `evaluate(track, t)`：查询 `t` 时刻的值。
- `copy_range(track, a, b)`：裁出长度 `b-a` 的独立 `Clip`，局部时间从 0 开始，
  端点可落在贝塞尔段内部（自动做 de Casteljau 分割）。
- `paste(track, d, e, clip, reverse=False)`：返回新轨道，将 clip 缩放到 `[d,e)`，
  区间外不变；`reverse=True` 时按左极限反向播放，内部跳变仍归属右侧片段。
  输入轨道与 clip 均不被修改。

## 测试

```
python3 -m pytest tests/ -q
```
