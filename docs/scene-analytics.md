# 场景分析使用说明

本阶段在现有 YOLO 和 ByteTrack 之后增加人数统计、车辆拥堵、车辆违停、消防通道
车辆/未知杂物占用和访客异常停留分析。摄像头不是必需条件，可以使用本地录像开发和验收。

## 生成场景配置

从录像首帧依次标注人员区域、拥堵区域、禁停区域、消防通道、访客停留监控区域和
出入口计数线：

```powershell
python tools/configure_scene.py --source data/gate.mp4 --output scenes/gate.yaml --scene-id gate_01
```

在标注窗口中，鼠标左键增加点，Backspace 撤销，Enter 确认当前区域，Esc 取消。
区域和线坐标以 `0～1` 归一化值写入 YAML。也可以复制 `scenes/example.yaml` 手动调整。

## 离线运行

```powershell
python main.py --source data/gate.mp4 --scene scenes/gate.yaml --no-show
```

不传 `--scene` 时，程序保持第一阶段行为，只执行检测、跟踪、视频和轨迹输出。

场景分析会额外生成：

- `metrics.jsonl`：按配置间隔输出当前人数、访客候选数、进出人数、车辆数和规则状态。
- `events.jsonl`：记录告警开始和结束，连续事件不会重复创建。
- `events/*.jpg`：告警开始时的画面证据。
- `summary.json`：最终人数统计和各类告警数量。

## 人工标注与评估

人工标注文件示例：

```json
{
  "counts": {"entries": 12, "exits": 7},
  "events": [
    {"event_type": "traffic_congestion", "region": "congestion_area", "start": 15.0, "end": 34.0},
    {"event_type": "illegal_parking", "region": "no_parking_area", "start": 48.0, "end": 82.0},
    {"event_type": "visitor_abnormal_stay", "region": "visitor_watch_area", "start": 90.0, "end": 132.0}
  ]
}
```

执行评估：

```powershell
python tools/evaluate.py --truth data/gate.labels.json --events outputs/events.jsonl --output outputs/evaluation.json
```

报告包含进出人数绝对误差、事件 Precision、Recall、F1 和平均告警延迟。规则阈值应使用
实际场景的正常通行、昼夜访客停留、持续拥堵、真实违停和消防通道占用录像分别调校。


## 消防通道未知杂物

先在通道畅通时生成基准图：

```powershell
python tools/capture_baseline.py --source data/gate.mp4 --output data/baselines/gate.jpg --duration-seconds 3
```

在场景 YAML 的 `rules.fire_lane_obstruction` 中配置消防通道区域、基准图片、像素差异、
最小占用面积、持续时间和恢复时间。检测器会用当前人车跟踪框清除动态目标，持续存在的
剩余区域作为未知障碍物；全局变化面积超过阈值时按光照或摄像头变化抑制，不产生杂物
告警。
