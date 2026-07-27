# 嵌入式安消智能软件 MVP

当前版本实现视频中的人员和车辆检测与多目标跟踪。系统使用 Ultralytics YOLO
预训练模型和 ByteTrack，对 `person`、`car`、`motorcycle`、`bus`、`truck`
五类目标持续分配 Track ID，并支持区域人数、进出人数、车辆拥堵、车辆违停、
消防通道车辆/未知杂物占用以及访客/未知人员异常停留分析。

## 环境准备

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

默认模型是 `yolo11n.pt`。首次运行时 Ultralytics 会自动下载模型；也可以把本地
`.pt` 权重路径写入 `config.yaml`。程序优先使用 CUDA，CUDA 不可用时自动使用 CPU。

## 运行

本地视频：

```powershell
python main.py --source data/test.mp4
```

摄像头：

```powershell
python main.py --source 0
```

RTSP：

```powershell
python main.py --source "rtsp://user:password@host:554/stream"
```

无窗口运行并限制处理帧数：

```powershell
python main.py --source data/test.mp4 --no-show --max-frames 100
```

常用覆盖参数：

```powershell
python main.py --source data/test.mp4 --model models/custom.pt --device cpu
python main.py --source data/test.mp4 --output-dir outputs/run-01 --no-save-video
```

在预览窗口按 `q` 可安全退出。视频源、模型、阈值、跟踪器、输出和重连参数均可在
`config.yaml` 中调整。

## 输出

默认输出目录为 `outputs/`：

- `tracked.mp4`：包含检测框、类别、Track ID、置信度、FPS 和实时数量的结果视频。
- `tracks.jsonl`：每行一个被跟踪目标，方便后续数据库或告警模块逐行消费。
- `metrics.jsonl`：场景人数、车辆数、异常停留候选数和活动事件等周期指标。
- `events.jsonl`：告警开始与结束事件，包括区域、Track ID、持续时间和身份信息。
- `events/*.jpg`：每个开始告警对应的证据截图。
- `summary.json`：处理帧数、各类事件数量和最终指标。

JSONL 示例：

```json
{"track_id":1,"class_id":0,"class_name":"person","confidence":0.912345,"bbox":[12.5,24.0,180.0,390.5],"frame_id":15,"timestamp":0.5}
```

`timestamp` 对视频文件表示视频内时间（秒），对摄像头或网络流表示本次运行开始后的
相对时间（秒）。当前人员数和车辆数指当前帧中仍被 ByteTrack 跟踪的目标数量。

## 项目结构

```text
main.py                    命令行入口和错误处理
config.yaml                默认运行配置
src/config.py              配置读取、路径解析和校验
src/video_source.py        文件、摄像头、RTSP 输入及重连
src/detector_tracker.py    YOLO + ByteTrack 推理和统一结果转换
src/identity/              人员身份接口和默认 unknown 身份提供器
src/rules/                 人数、拥堵、违停、消防通道和异常停留规则
src/obstruction/           基准背景差分、动态目标排除和障碍区域提取
src/analytics/             轨迹历史、事件指标、叠加显示和分析输出
src/scene_config.py        区域、昼夜时段和规则配置
src/visualizer.py          检测框、状态信息和计数绘制
src/result_writer.py       MP4 与 JSONL 输出
src/application.py         端到端处理主循环
src/schemas.py             帧和跟踪结果数据结构
tests/                     不依赖模型下载的自动化测试
```

## 测试

单元测试不需要下载模型：

```powershell
python -m unittest discover -s tests -v
```

完整冒烟测试需要一段本地视频和模型权重：

```powershell
python main.py --source data/test.mp4 --no-show --max-frames 30
```

验收时应确认输出视频可以播放、同一连续目标的 Track ID 基本稳定、JSONL 字段完整，
并分别记录 CPU/CUDA 设备、视频分辨率、总帧数、耗时和平均 FPS。

## 场景分析

通过场景 YAML 可以启用指定区域人数、进出人数、车辆拥堵、车辆违停、消防通道车辆/
未知杂物占用和访客异常停留分析。先从录像首帧生成区域配置：

```powershell
python tools/configure_scene.py --source data/test.mp4 --output scenes/test.yaml --scene-id gate_01
```

然后使用本地录像离线运行：

```powershell
python main.py --source data/test.mp4 --scene scenes/test.yaml --no-show
```

除 `tracked.mp4` 和 `tracks.jsonl` 外，还会生成 `metrics.jsonl`、`events.jsonl`、
`events/*.jpg` 和 `summary.json`。详细配置和人工标注评估方法见
[`docs/scene-analytics.md`](docs/scene-analytics.md)。

## 消防通道未知杂物占用

该功能不需要额外训练模型。固定摄像头画面与畅通状态基准图进行差分，当前 YOLO
跟踪结果用于排除人员和车辆；剩余变化区域持续达到阈值后生成
`fire_lane_obstruction` 告警。车辆仍由原有 `fire_lane_occupied` 规则处理。

生成基准图并运行 ABODA 实拍遗留物验证：

```powershell
python tools/capture_baseline.py --source data\public\aboda_video1.avi --output data\baselines\aboda_video1.jpg --duration-seconds 1
python main.py --source data\public\aboda_video1.avi --scene scenes\aboda_obstruction.yaml --output-dir outputs\verify-obstruction --no-show
```

场景规则支持 `pixel_threshold`、`min_area_ratio`、`hold_seconds`、
`recovery_seconds`、人员车辆框扩展比例和全局画面变化抑制阈值。测试视频使用3秒触发，
现场建议从20至60秒开始标定。摄像头必须固定；基准图应在通道完全畅通时采集。

## 访客异常停留

功能 6 已实现为可插拔身份模式。当前默认使用 `UnknownIdentityProvider`，将检测到但
尚未完成人脸识别的人员标记为 `unknown`；规则可以同时监控 `visitor` 和
`unknown`。后续接入人脸识别时，只需实现 `IdentityProvider`，停留规则和事件格式
不需要修改。

场景配置示例：

```yaml
regions:
  visitor_watch_area:
    polygon: [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]

rules:
  visitor_loitering:
    enabled: true
    include_roles: [visitor, unknown]
    period: auto
    day_start: "06:00"
    night_start: "18:00"
    zones:
      - region: visitor_watch_area
        day_hold_seconds: 120
        night_hold_seconds: 30
        absence_grace_seconds: 3
        recovery_seconds: 2
```

`period: auto` 使用边缘设备本地时间选择昼夜阈值；离线录像无法确定真实拍摄时间时，
应明确设置为 `day` 或 `night`。人员在区域内可以走动，规则按连续在场时间判断，
短时遮挡在 `absence_grace_seconds` 内不会清零。

使用当前实拍行人素材进行快速验证：

```powershell
python main.py --source data\public\opencv_vtest.avi --scene scenes\visitor_loitering_example.yaml --output-dir outputs\verify-loitering --no-show
```

达到阈值后会生成 `visitor_abnormal_stay` 开始事件和证据截图；人员离开区域并超过
恢复时间后生成同一 `event_id` 的结束事件。指标中包含
`current_visitors`、`visitor_loitering_pending`、
`visitor_loitering_track_ids` 和 `visitor_loitering_regions`。

## 当前边界

本阶段已支持基于本地录像或视频流的人员统计、车辆拥堵、车辆违停、消防通道车辆/
未知杂物占用和访客/未知人员异常停留规则。身份接口已经建立，但暂未实现人脸特征提取、
员工访客库和跨摄像头 ReID；因此默认身份只能视为待识别人员，不能作为最终身份结论。

当前仍不包含模型训练、车牌识别、杂物类别识别、数据库、智慧终端上传、前端或微服务。
遮挡超过容忍时间、目标长时间离开画面以及没有稳定身份时的 Track ID 切换仍可能重置
停留计时。所有规则阈值和区域必须使用目标园区录像标定后再用于验收。

## 电脑与 RK3588 双后端

项目只维护一套规则代码。电脑端由 `config.yaml` 选择
`ultralytics + yolo11n.pt`，RK3588 由 `config.rk3588.yaml` 选择
`RKNNLite + yolo11n_fp16.rknn`。两种后端都先生成统一检测结果，再由共享的
两阶段跟踪器生成 `TrackResult`，因此人数、拥堵、违停、消防通道占用、杂物和
异常停留规则不区分运行设备。

电脑端安装和运行：

```powershell
python -m pip install -r requirements.txt
python main.py --config config.yaml --source data\public\opencv_vtest.avi --scene scenes\public_vtest_real.yaml --no-show
```

RKNN实现位于 `src/inference/rknn_detector.py`，输入是已经在板上验证过的
`(1, 640, 640, 3) uint8`，输出按 `(1, 84, 8400)` 解码，并完成类别过滤、
置信度过滤、逐类别 NMS 和 letterbox 坐标还原。

## 生成 RK3588 部署包

默认使用已经导出的 FP16 模型：

```powershell
python tools\build_rk3588_bundle.py --version 0.1.0
```

生成文件：

```text
dist/park-safety-rk3588-0.1.0.tar.gz
```

部署包只包含运行代码、场景配置、RKNN模型和启动脚本，不包含 `.git`、
PyTorch、Ultralytics、公开测试视频、历史输出、PPT或测试代码。指定其他模型时：

```powershell
python tools\build_rk3588_bundle.py --model D:\path\model.rknn --version 0.1.1
```

## 在 RK3588 上测试

先把部署包和测试视频传到板子：

```powershell
scp dist\park-safety-rk3588-0.1.0.tar.gz root@板子IP:/root/
scp data\public\opencv_vtest.avi root@板子IP:/root/
```

登录板子后解压并检查环境：

```bash
cd /root
tar -xzf park-safety-rk3588-0.1.0.tar.gz
cd park-safety-rk3588-0.1.0

python3 -c "import cv2, numpy, yaml; from rknnlite.api import RKNNLite; print('runtime OK')"
```

如果缺少 OpenCV、NumPy 或 PyYAML，可联网安装
`python3 -m pip install -r requirements.txt`；`rknn-toolkit-lite2` 使用板子中
已经安装的 2.3.2 版本，不要用电脑端依赖覆盖。

先验证一帧检测结果：

```bash
python3 tools/rknn_smoke_test.py \
  --model models/yolo11n_fp16.rknn \
  --source /root/opencv_vtest.avi \
  --output outputs/rknn-smoke.jpg
```

再运行完整视频和业务规则：

```bash
./start.sh /root/opencv_vtest.avi scenes/public_vtest_real.yaml
```

结果位于 `outputs/`，重点检查 `tracked.mp4`、`events.jsonl`、
`metrics.jsonl`、`summary.json` 和 `events/*.jpg`。

长期运行时执行：

```bash
sudo ./install.sh
sudo nano /etc/park-safety/park-safety.env
sudo systemctl start park-safety
sudo journalctl -u park-safety -f
```

在 `park-safety.env` 中配置 `VIDEO_SOURCE` 和实际的 `SCENE_CONFIG`。
没有摄像头时使用板上的视频路径；以后接摄像头只需要将 `VIDEO_SOURCE` 改为
摄像头编号或 RTSP 地址，不需要修改算法代码。
