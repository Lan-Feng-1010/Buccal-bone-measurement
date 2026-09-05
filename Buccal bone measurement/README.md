# 术后 CBCT 三维分割测量工具 v0.3.0

从**已有三维 NIfTI 分割**开始，在种植体平台向根方 0、2、4、6 mm 处，测量**种植体中轴线到颊侧骨外缘**的距离，输出数值、三维端点和对应剖面图。

本版本不包含二维分割、截图轮廓推理或医生 JSON 标注测量功能。医生截图和人工标注属于后续独立验证所用的参考资料，不参与本算法运行。

## 1. 流程与当前能力

匿名 CBCT NIfTI → 外部分割程序生成三维标签 → 本工具提取目标种植体和颌骨 → 使用平台、中轴线与牙弓方向确定剖面 → 在三维分割中计算四个高度的测量射线 → 输出测量及剖面可视化。

**测量直接在三维体素和 NIfTI 物理坐标中完成。** 剖面图是三维数据重采样后的展示结果，不是另一次分割，也不是测量输入。其显示采样间距不会替代原始体素间距，不会提升原始影像分辨率。

当前是研究原型：已有三维分割可直接使用；本包不含 ToothFairy3 / ToothSeg 推理程序和模型权重。平台、朝根方的中轴线和颊侧方向仍需调用者提供可靠信息。SVD 仅提供位姿候选，不能当作已经验证的自动平台识别；多种植体时需明确选择目标。当前尚未实现无需这些信息即可完成的全自动流程。

## 2. 安装与合成示例

需要 Python 3.11 或更高版本。解压后进入本 README 所在目录。

Windows：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m postop_cbct demo --output demo_output --plot
```

macOS / Linux：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m postop_cbct demo --output demo_output --plot
```

合成体模的四个预期距离均为 **4.75 mm**，仅用于验证程序运行。它不是临床病例，不能证明临床准确性。后文 `python` 均指已安装依赖的解释器。

如需在其他目录调用，可安装本项目：

```sh
python -m pip install -e ".[plot]"
postop-cbct --help
```

也可安装 `dist` 中的 wheel 文件。依赖不随压缩包分发；安装依赖需要网络或预先准备的离线依赖包。不绘图时可省略 `--plot`，无需 matplotlib。

## 3. 使用已有三维二值分割

复制 `examples/binary_masks.template.json`，填入自己的文件相对路径及真实几何参数。模板中 `null` 为待填写信息，不是有效的自动定位结果。

```sh
python -m postop_cbct measure examples/binary_masks.template.json --output results/CASE001 --plot
```

输入要求：

- `target_implant_mask_nifti`：仅含一个目标种植体的三维 0/1 掩膜。
- `jaw_mask_nifti`：对应颌骨的三维 0/1 掩膜。
- 两者 shape、affine 必须一致，空间单位必须为 mm；不执行隐式配准或重采样。
- 若分割单位为 unknown，必须提供同 shape、同 affine 且明确采用 mm 的 `reference_cbct_nifti`，核对后才能继承单位。
- `platform_ras`：平台中心的三维 RAS 坐标，单位 mm。
- `axis_apical_ras`：从平台指向根方的三维方向向量。
- `arch_tangent_ras`：种植体所在位置的牙弓切线方向；或者提供至少三个同颌牙中心 `same_jaw_tooth_centers_ras` 供拟合。默认不接受参考牙范围外的外推。
- `buccal_hint_ras`：指向颊侧的三维方向向量，用于确定测量方向的正负。

NIfTI 使用 RAS 物理坐标，不能把医生截图“上颌平台在图像下方、下颌平台在上方”的显示规则直接套用到数组索引。截图固定的 0.1 mm/pixel 不参与三维测量标定。

## 4. 使用 ToothFairy3 / ToothSeg 三维多标签分割

合作方只有匿名 CBCT 时，应先在其分割环境生成三维标签。可采用以下目录结构，具体路径可自行修改：

```text
data/CASE001/
  cbct.nii.gz
  toothfairy3.nii.gz
  toothseg.nii.gz
```

复制并填写 `examples/full_segmentation.template.json`，先检查种植体连通分量：

```sh
python -m postop_cbct inspect examples/full_segmentation.template.json --output results/components.json
```

有多个候选时填写 `target_component_id`；程序不会默认选择体积最大者。连通分量编号不等于 FDI 牙位，`target_fdi` 仅记录调用者提供的信息。确认 `jaw` 以及平台、中轴线和颊侧方向后运行：

```sh
python -m postop_cbct prepare examples/full_segmentation.template.json --output results/prepared
python -m postop_cbct measure results/prepared/request.json --output results/measured --plot
```

默认 ToothFairy3 标签为 1=下颌骨、2=上颌骨、10=种植体，只能在确认分割采用这一标签体系时使用。可通过 `jaw_label`、`implant_label` 覆盖。

ToothSeg 默认标签 1–8→21–28，9–16→11–18，17–24→41–48，25–32→31–38。可通过 `toothseg_label_to_fdi` 自定义。程序提取同颌牙中心，供牙弓拟合使用；不等于自动确认目标种植体牙位。已有牙弓方向或牙中心时，可不提供 ToothSeg。

`prepare` 输出目标种植体、颌骨二值 NIfTI、`request.json` 和准备记录。它不复制原始 CBCT；如需灰度剖面背景，请在生成的 `request.json` 中加入正确的 `reference_cbct_nifti`。

## 5. 输出与路径约定

`measurement_3d.json` 包括各高度的距离、三维测量端点、剖面坐标变换和失败原因。`segmentation_slice.npy` 是**已有三维分割的剖面采样**；提供原始 CBCT 时另有 `cbct_slice.npy`。这些文件均为结果，不是二维分割模型的输入或预测。`--plot` 生成 PNG、PDF、SVG 测量图。

相对输入路径以**配置 JSON 所在目录**为基准；相对 `--output` 路径以当前命令目录为基准。代码不依赖作者的盘符或病例路径。配置也可使用接收者自己的绝对路径。输出结果不记录原始文件完整路径，本地报错可能显示路径以便排查。不同病例请使用不同输出目录，避免同名结果被覆盖。

## 6. 结果解释与验证

无骨、多段骨、边界被视野或搜索范围截断、测量起点不在种植体内等情况，返回空值或复核原因，不用 0 代替失败。测量值不减种植体半径，也不是单独的骨板厚度。`automatic_release` 始终为 `false`，数值计算成功不等于已通过临床复核。

```sh
python -m unittest discover -s tests -v
```

验证内容和适用边界见 `VALIDATION.md`，几何定义见 `docs/METHODS.md`。医生截图与 JSON 可在独立的研究评估流程中作为参考标准；本包不加载或解析这些人工标注。

## 7. v0.3.0 修改

移除了 `annotate2d`、`measure2d` 命令及其标注适配器、二维测量模块和对应测试。保留三维几何测量和由三维数据生成的剖面展示；不改变四个高度及距离定义。请用本版本替换旧分享包。
