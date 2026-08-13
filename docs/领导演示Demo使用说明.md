# 园区安消领导演示 Demo

本 Demo 展示三项已经具备完整处理链路的能力：人员进出统计、出入口车辆拥堵预警、公共区域堆物预警。

## 一键生成

在项目目录执行：

```powershell
python tools/run_demo_suite.py
```

生成结果位于：

- `outputs/leadership-demo/people/result.mp4`
- `outputs/leadership-demo/congestion/result.mp4`
- `outputs/leadership-demo/clutter/result.mp4`

脚本还会在 `outputs/leadership-demo/ppt/` 中生成三段适合直接插入 PowerPoint 的 H.264 MP4。三段视频分别为 17 秒、12 秒和 14.5 秒，均采用 1280×720、H.264 High Profile、YUV 4:2:0，并启用快速加载标记。

只生成某一项时使用 `--only people`、`--only congestion` 或 `--only clutter`。需要边处理边预览时增加 `--show`。

## 画面说明

人员统计画面显示当前人数、进入人数和离开人数；车辆拥堵画面显示监测区域内车辆数和低速车辆比例；公共区域堆物画面显示相对空场基准图的持续变化面积。达到规则阈值后，监测区域变红，状态变为“报警”，底部显示紧凑的中文报警条。

演示配置独立放在 `config.demo.yaml` 和 `scenes/demo/` 中。其中的持续时间为便于短视频演示而缩短，现场部署前应根据摄像头视角、道路尺度和管理规则重新标定，不能直接作为生产阈值。

中文叠加文字会优先使用 Windows 微软雅黑或 Linux Noto CJK 字体。边缘设备如未安装中文字体，可通过 `PARK_SAFETY_FONT` 和 `PARK_SAFETY_BOLD_FONT` 指定常规与粗体字体文件。
