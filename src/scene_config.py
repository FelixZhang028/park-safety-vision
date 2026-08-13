from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import ConfigurationError
from .identity.schemas import IDENTITY_ROLES
from .spatial.geometry import Point, distance


COUNT_DIRECTIONS = {"negative_to_positive", "positive_to_negative"}
LOITERING_PERIODS = {"auto", "day", "night"}


@dataclass(slots=True, frozen=True)
class RegionConfig:
    name: str
    polygon: tuple[Point, ...]


@dataclass(slots=True, frozen=True)
class CountingLineConfig:
    name: str
    points: tuple[Point, Point]
    in_direction: str = "negative_to_positive"
    hysteresis: float = 0.01
    cooldown_seconds: float = 1.0


@dataclass(slots=True, frozen=True)
class PersonCountRuleConfig:
    enabled: bool = False
    region: str | None = None
    lines: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class CongestionRuleConfig:
    enabled: bool = False
    region: str | None = None
    min_vehicles: int = 4
    max_speed_ratio: float = 0.01
    min_low_speed_ratio: float = 0.70
    hold_seconds: float = 10.0
    recovery_seconds: float = 5.0


@dataclass(slots=True, frozen=True)
class IllegalParkingRuleConfig:
    enabled: bool = False
    region: str | None = None
    max_speed_ratio: float = 0.005
    hold_seconds: float = 30.0
    recovery_seconds: float = 2.0
    suppress_when_congested: bool = True


@dataclass(slots=True, frozen=True)
class FireLaneRuleConfig:
    enabled: bool = False
    region: str | None = None
    max_speed_ratio: float = 0.005
    hold_seconds: float = 10.0
    recovery_seconds: float = 2.0


@dataclass(slots=True, frozen=True)
class FireLaneObstructionRuleConfig:
    enabled: bool = False
    region: str | None = None
    baseline_path: Path | None = None
    pixel_threshold: int = 28
    min_area_ratio: float = 0.002
    hold_seconds: float = 30.0
    recovery_seconds: float = 5.0
    exclusion_padding_ratio: float = 0.02
    max_global_change_ratio: float = 0.35


@dataclass(slots=True, frozen=True)
class VisitorLoiteringZoneConfig:
    region: str
    day_hold_seconds: float = 120.0
    night_hold_seconds: float = 30.0
    absence_grace_seconds: float = 3.0
    recovery_seconds: float = 2.0


@dataclass(slots=True, frozen=True)
class VisitorLoiteringRuleConfig:
    enabled: bool = False
    include_roles: tuple[str, ...] = ("visitor", "unknown")
    period: str = "auto"
    day_start: str = "06:00"
    night_start: str = "18:00"
    zones: tuple[VisitorLoiteringZoneConfig, ...] = ()


@dataclass(slots=True, frozen=True)
class SceneConfig:
    scene_id: str
    regions: dict[str, RegionConfig]
    lines: dict[str, CountingLineConfig]
    display_name: str = ""
    location: str = ""
    person_count: PersonCountRuleConfig = field(default_factory=PersonCountRuleConfig)
    congestion: CongestionRuleConfig = field(default_factory=CongestionRuleConfig)
    illegal_parking: IllegalParkingRuleConfig = field(
        default_factory=IllegalParkingRuleConfig
    )
    fire_lane: FireLaneRuleConfig = field(default_factory=FireLaneRuleConfig)
    fire_lane_obstruction: FireLaneObstructionRuleConfig = field(
        default_factory=FireLaneObstructionRuleConfig
    )
    visitor_loitering: VisitorLoiteringRuleConfig = field(
        default_factory=VisitorLoiteringRuleConfig
    )
    history_seconds: float = 2.0
    stale_track_seconds: float = 2.0
    metrics_interval_seconds: float = 1.0

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SceneConfig":
        scene_path = Path(path).expanduser().resolve()
        if not scene_path.is_file():
            raise ConfigurationError(f"Scene file does not exist: {scene_path}")
        try:
            raw = yaml.safe_load(scene_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Cannot read scene configuration: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ConfigurationError("The scene configuration root must be a mapping")
        return cls.from_mapping(raw, base_directory=scene_path.parent)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        base_directory: Path | None = None,
    ) -> "SceneConfig":
        _reject_unknown(
            raw, {"scene", "regions", "lines", "rules", "analytics"}, "scene root"
        )
        scene_raw = _mapping(raw.get("scene", {}), "scene")
        regions_raw = _mapping(raw.get("regions", {}), "regions")
        lines_raw = _mapping(raw.get("lines", {}), "lines")
        rules_raw = _mapping(raw.get("rules", {}), "rules")
        analytics_raw = _mapping(raw.get("analytics", {}), "analytics")

        _reject_unknown(scene_raw, {"id", "name", "location"}, "scene")
        _reject_unknown(
            rules_raw,
            {
                "person_count",
                "congestion",
                "illegal_parking",
                "fire_lane",
                "fire_lane_obstruction",
                "visitor_loitering",
            },
            "rules",
        )
        _reject_unknown(
            analytics_raw,
            {"history_seconds", "stale_track_seconds", "metrics_interval_seconds"},
            "analytics",
        )

        regions = {
            str(name): _parse_region(str(name), value)
            for name, value in regions_raw.items()
        }
        lines = {
            str(name): _parse_line(str(name), value)
            for name, value in lines_raw.items()
        }
        config = cls(
            scene_id=str(scene_raw.get("id", "scene")),
            regions=regions,
            lines=lines,
            display_name=str(scene_raw.get("name", "")).strip(),
            location=str(scene_raw.get("location", "")).strip(),
            person_count=_parse_person_count(rules_raw.get("person_count", {})),
            congestion=_parse_congestion(rules_raw.get("congestion", {})),
            illegal_parking=_parse_illegal_parking(
                rules_raw.get("illegal_parking", {})
            ),
            fire_lane=_parse_fire_lane(rules_raw.get("fire_lane", {})),
            fire_lane_obstruction=_parse_fire_lane_obstruction(
                rules_raw.get("fire_lane_obstruction", {}),
                base_directory,
            ),
            visitor_loitering=_parse_visitor_loitering(
                rules_raw.get("visitor_loitering", {})
            ),
            history_seconds=float(analytics_raw.get("history_seconds", 2.0)),
            stale_track_seconds=float(analytics_raw.get("stale_track_seconds", 2.0)),
            metrics_interval_seconds=float(
                analytics_raw.get("metrics_interval_seconds", 1.0)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.scene_id.strip():
            raise ConfigurationError("scene.id cannot be empty")
        if self.history_seconds <= 0 or self.stale_track_seconds <= 0:
            raise ConfigurationError(
                "analytics history values must be greater than zero"
            )
        if self.metrics_interval_seconds <= 0:
            raise ConfigurationError(
                "analytics.metrics_interval_seconds must be greater than zero"
            )

        if self.person_count.enabled:
            self._require_region(self.person_count.region, "person_count")
            missing_lines = sorted(set(self.person_count.lines) - set(self.lines))
            if missing_lines:
                raise ConfigurationError(
                    f"person_count references unknown line(s): {', '.join(missing_lines)}"
                )
        if self.congestion.enabled:
            self._require_region(self.congestion.region, "congestion")
            if self.congestion.min_vehicles <= 0:
                raise ConfigurationError(
                    "congestion.min_vehicles must be greater than zero"
                )
            if not 0 <= self.congestion.min_low_speed_ratio <= 1:
                raise ConfigurationError(
                    "congestion.min_low_speed_ratio must be between 0 and 1"
                )
        if self.illegal_parking.enabled:
            self._require_region(self.illegal_parking.region, "illegal_parking")
        if self.fire_lane.enabled:
            self._require_region(self.fire_lane.region, "fire_lane")
        if self.fire_lane_obstruction.enabled:
            self._require_region(
                self.fire_lane_obstruction.region,
                "fire_lane_obstruction",
            )
            if self.fire_lane_obstruction.baseline_path is None:
                raise ConfigurationError(
                    "fire_lane_obstruction.baseline_path cannot be empty"
                )
        if self.visitor_loitering.enabled:
            if not self.visitor_loitering.zones:
                raise ConfigurationError(
                    "visitor_loitering.zones must contain at least one zone"
                )
            for zone in self.visitor_loitering.zones:
                self._require_region(zone.region, "visitor_loitering")

        if self.visitor_loitering.period not in LOITERING_PERIODS:
            raise ConfigurationError(
                "visitor_loitering.period must be one of: auto, day, night"
            )
        invalid_roles = sorted(
            set(self.visitor_loitering.include_roles) - IDENTITY_ROLES
        )
        if not self.visitor_loitering.include_roles or invalid_roles:
            raise ConfigurationError(
                "visitor_loitering.include_roles must use: employee, visitor, unknown"
            )
        day_start = clock_minutes(self.visitor_loitering.day_start)
        night_start = clock_minutes(self.visitor_loitering.night_start)
        if day_start == night_start:
            raise ConfigurationError(
                "visitor_loitering day_start and night_start must be different"
            )
        for zone in self.visitor_loitering.zones:
            values = (
                zone.day_hold_seconds,
                zone.night_hold_seconds,
                zone.absence_grace_seconds,
                zone.recovery_seconds,
            )
            if any(value < 0 for value in values):
                raise ConfigurationError(
                    "visitor_loitering zone durations cannot be negative"
                )

        durations = (
            self.congestion.hold_seconds,
            self.congestion.recovery_seconds,
            self.illegal_parking.hold_seconds,
            self.illegal_parking.recovery_seconds,
            self.fire_lane.hold_seconds,
            self.fire_lane.recovery_seconds,
            self.fire_lane_obstruction.hold_seconds,
            self.fire_lane_obstruction.recovery_seconds,
        )
        if any(value < 0 for value in durations):
            raise ConfigurationError("rule hold and recovery values cannot be negative")
        speeds = (
            self.congestion.max_speed_ratio,
            self.illegal_parking.max_speed_ratio,
            self.fire_lane.max_speed_ratio,
        )
        if any(value < 0 for value in speeds):
            raise ConfigurationError("rule speed ratios cannot be negative")
        obstruction = self.fire_lane_obstruction
        if not 0 <= obstruction.pixel_threshold <= 255:
            raise ConfigurationError(
                "fire_lane_obstruction.pixel_threshold must be between 0 and 255"
            )
        if not 0 < obstruction.min_area_ratio <= 1:
            raise ConfigurationError(
                "fire_lane_obstruction.min_area_ratio must be between 0 and 1"
            )
        if not 0 < obstruction.max_global_change_ratio <= 1:
            raise ConfigurationError(
                "fire_lane_obstruction.max_global_change_ratio must be between 0 and 1"
            )
        if not 0 <= obstruction.exclusion_padding_ratio <= 0.5:
            raise ConfigurationError(
                "fire_lane_obstruction.exclusion_padding_ratio must be between "
                "0 and 0.5"
            )

    def _require_region(self, region: str | None, rule_name: str) -> None:
        if not region or region not in self.regions:
            raise ConfigurationError(
                f"{rule_name} references unknown region: {region or '<empty>'}"
            )


def _parse_region(name: str, value: Any) -> RegionConfig:
    raw = _mapping(value, f"regions.{name}")
    _reject_unknown(raw, {"polygon"}, f"regions.{name}")
    polygon = _points(raw.get("polygon", []), f"regions.{name}.polygon")
    if len(polygon) < 3:
        raise ConfigurationError(f"regions.{name}.polygon needs at least 3 points")
    return RegionConfig(name=name, polygon=polygon)


def _parse_line(name: str, value: Any) -> CountingLineConfig:
    raw = _mapping(value, f"lines.{name}")
    _reject_unknown(
        raw,
        {"points", "in_direction", "hysteresis", "cooldown_seconds"},
        f"lines.{name}",
    )
    points = _points(raw.get("points", []), f"lines.{name}.points")
    if len(points) != 2 or distance(points[0], points[1]) <= 1e-9:
        raise ConfigurationError(f"lines.{name}.points must contain 2 different points")
    direction = str(raw.get("in_direction", "negative_to_positive"))
    if direction not in COUNT_DIRECTIONS:
        raise ConfigurationError(
            f"lines.{name}.in_direction must be one of: {', '.join(sorted(COUNT_DIRECTIONS))}"
        )
    hysteresis = float(raw.get("hysteresis", 0.01))
    cooldown = float(raw.get("cooldown_seconds", 1.0))
    if hysteresis < 0 or cooldown < 0:
        raise ConfigurationError(f"lines.{name} thresholds cannot be negative")
    return CountingLineConfig(
        name=name,
        points=(points[0], points[1]),
        in_direction=direction,
        hysteresis=hysteresis,
        cooldown_seconds=cooldown,
    )


def _parse_person_count(value: Any) -> PersonCountRuleConfig:
    raw = _mapping(value, "rules.person_count")
    _reject_unknown(raw, {"enabled", "region", "lines"}, "rules.person_count")
    return PersonCountRuleConfig(
        enabled=bool(raw.get("enabled", bool(raw))),
        region=_optional_string(raw.get("region")),
        lines=tuple(str(name) for name in raw.get("lines", [])),
    )


def _parse_congestion(value: Any) -> CongestionRuleConfig:
    raw = _mapping(value, "rules.congestion")
    _reject_unknown(
        raw,
        {
            "enabled",
            "region",
            "min_vehicles",
            "max_speed_ratio",
            "min_low_speed_ratio",
            "hold_seconds",
            "recovery_seconds",
        },
        "rules.congestion",
    )
    return CongestionRuleConfig(
        enabled=bool(raw.get("enabled", bool(raw))),
        region=_optional_string(raw.get("region")),
        min_vehicles=int(raw.get("min_vehicles", 4)),
        max_speed_ratio=float(raw.get("max_speed_ratio", 0.01)),
        min_low_speed_ratio=float(raw.get("min_low_speed_ratio", 0.70)),
        hold_seconds=float(raw.get("hold_seconds", 10.0)),
        recovery_seconds=float(raw.get("recovery_seconds", 5.0)),
    )


def _parse_illegal_parking(value: Any) -> IllegalParkingRuleConfig:
    raw = _mapping(value, "rules.illegal_parking")
    _reject_unknown(
        raw,
        {
            "enabled",
            "region",
            "max_speed_ratio",
            "hold_seconds",
            "recovery_seconds",
            "suppress_when_congested",
        },
        "rules.illegal_parking",
    )
    return IllegalParkingRuleConfig(
        enabled=bool(raw.get("enabled", bool(raw))),
        region=_optional_string(raw.get("region")),
        max_speed_ratio=float(raw.get("max_speed_ratio", 0.005)),
        hold_seconds=float(raw.get("hold_seconds", 30.0)),
        recovery_seconds=float(raw.get("recovery_seconds", 2.0)),
        suppress_when_congested=bool(raw.get("suppress_when_congested", True)),
    )


def _parse_fire_lane(value: Any) -> FireLaneRuleConfig:
    raw = _mapping(value, "rules.fire_lane")
    _reject_unknown(
        raw,
        {
            "enabled",
            "region",
            "max_speed_ratio",
            "hold_seconds",
            "recovery_seconds",
        },
        "rules.fire_lane",
    )
    return FireLaneRuleConfig(
        enabled=bool(raw.get("enabled", bool(raw))),
        region=_optional_string(raw.get("region")),
        max_speed_ratio=float(raw.get("max_speed_ratio", 0.005)),
        hold_seconds=float(raw.get("hold_seconds", 10.0)),
        recovery_seconds=float(raw.get("recovery_seconds", 2.0)),
    )


def _parse_fire_lane_obstruction(
    value: Any,
    base_directory: Path | None,
) -> FireLaneObstructionRuleConfig:
    raw = _mapping(value, "rules.fire_lane_obstruction")
    _reject_unknown(
        raw,
        {
            "enabled",
            "region",
            "baseline_path",
            "pixel_threshold",
            "min_area_ratio",
            "hold_seconds",
            "recovery_seconds",
            "exclusion_padding_ratio",
            "max_global_change_ratio",
        },
        "rules.fire_lane_obstruction",
    )
    return FireLaneObstructionRuleConfig(
        enabled=bool(raw.get("enabled", bool(raw))),
        region=_optional_string(raw.get("region")),
        baseline_path=_optional_path(raw.get("baseline_path"), base_directory),
        pixel_threshold=int(raw.get("pixel_threshold", 28)),
        min_area_ratio=float(raw.get("min_area_ratio", 0.002)),
        hold_seconds=float(raw.get("hold_seconds", 30.0)),
        recovery_seconds=float(raw.get("recovery_seconds", 5.0)),
        exclusion_padding_ratio=float(raw.get("exclusion_padding_ratio", 0.02)),
        max_global_change_ratio=float(
            raw.get("max_global_change_ratio", 0.35)
        ),
    )


def _parse_visitor_loitering(value: Any) -> VisitorLoiteringRuleConfig:
    raw = _mapping(value, "rules.visitor_loitering")
    _reject_unknown(
        raw,
        {
            "enabled",
            "include_roles",
            "period",
            "day_start",
            "night_start",
            "zones",
        },
        "rules.visitor_loitering",
    )
    roles_raw = raw.get("include_roles", ["visitor", "unknown"])
    if not isinstance(roles_raw, (list, tuple)):
        raise ConfigurationError(
            "rules.visitor_loitering.include_roles must be a list"
        )
    zones_raw = raw.get("zones", [])
    if not isinstance(zones_raw, list):
        raise ConfigurationError("rules.visitor_loitering.zones must be a list")
    return VisitorLoiteringRuleConfig(
        enabled=bool(raw.get("enabled", bool(raw))),
        include_roles=tuple(str(role).lower() for role in roles_raw),
        period=str(raw.get("period", "auto")).lower(),
        day_start=str(raw.get("day_start", "06:00")),
        night_start=str(raw.get("night_start", "18:00")),
        zones=tuple(
            _parse_visitor_loitering_zone(index, zone)
            for index, zone in enumerate(zones_raw)
        ),
    )


def _parse_visitor_loitering_zone(
    index: int, value: Any
) -> VisitorLoiteringZoneConfig:
    field_name = f"rules.visitor_loitering.zones[{index}]"
    raw = _mapping(value, field_name)
    _reject_unknown(
        raw,
        {
            "region",
            "day_hold_seconds",
            "night_hold_seconds",
            "absence_grace_seconds",
            "recovery_seconds",
        },
        field_name,
    )
    region = str(raw.get("region", "")).strip()
    if not region:
        raise ConfigurationError(f"{field_name}.region cannot be empty")
    return VisitorLoiteringZoneConfig(
        region=region,
        day_hold_seconds=float(raw.get("day_hold_seconds", 120.0)),
        night_hold_seconds=float(raw.get("night_hold_seconds", 30.0)),
        absence_grace_seconds=float(raw.get("absence_grace_seconds", 3.0)),
        recovery_seconds=float(raw.get("recovery_seconds", 2.0)),
    )


def clock_minutes(value: str) -> int:
    parts = value.split(":")
    if (
        len(parts) != 2
        or not all(part.isdigit() for part in parts)
        or len(parts[0]) != 2
        or len(parts[1]) != 2
    ):
        raise ConfigurationError(f"Invalid clock time '{value}'; expected HH:MM")
    hour, minute = (int(part) for part in parts)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ConfigurationError(f"Invalid clock time '{value}'; expected HH:MM")
    return hour * 60 + minute


def _points(value: Any, field_name: str) -> tuple[Point, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{field_name} must be a list")
    points: list[Point] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigurationError(f"{field_name}[{index}] must be [x, y]")
        point = float(item[0]), float(item[1])
        if not all(0.0 <= coordinate <= 1.0 for coordinate in point):
            raise ConfigurationError(
                f"{field_name}[{index}] must be normalized to 0..1"
            )
        points.append(point)
    return tuple(points)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_name} must be a mapping")
    return value


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_path(value: Any, base_directory: Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_directory is not None:
        path = base_directory / path
    return path.resolve()


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration key(s) in {field_name}: {', '.join(unknown)}"
        )
