# keyframe_clip

内存中的关键帧曲线剪贴库（单标量动画轨道），支持裁取、变速粘贴和反向粘贴。

## 数据模型

- `Track(segments)`：覆盖 `[0, T)` 的片段元组，连续且不重叠；每段含左端、不含右端。
- 片段类型：`Hold(t0, t1, v0)`、`Linear(t0, t1, v0, v1)`、
  `Bezier(t0, t1, v0, v1, x1, y1, x2, y2)`（二维三次贝塞尔，`t0<=x1<=x2<=t1`，纵向可过冲）。
- `Clip` 即局部时间从 0 开始的 `Track`。

## 接口

- `evaluate(track, t)`：查询 t 时刻的值。
- `copy_range(track, a, b)`：裁出 `[a, b)` 的独立 clip，长度为 `b-a`。
- `paste(track, d, e, clip, reverse=False)`：把 clip 缩放到 `[d, e)` 生成新轨道，
  区间外不变；`reverse=True` 时按 `clip(L-s)` 的左极限取值。输入均不被修改。

## 测试

```sh
python3 -m pytest test_keyframe_clip.py -q
```
