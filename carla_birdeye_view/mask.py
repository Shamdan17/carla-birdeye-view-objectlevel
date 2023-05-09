import carla
import numpy as np

from typing import NamedTuple, List, Tuple, Optional
from carla_birdeye_view import lanes
from cv2 import cv2 as cv
from carla_birdeye_view.lanes import LaneSide

Mask = np.ndarray  # of shape (y, x), stores 0 and 1, dtype=np.int32
RoadSegmentWaypoints = List[carla.Waypoint]

COLOR_OFF = 0
COLOR_ON = 1


class Coord(NamedTuple):
    x: int
    y: int


class FltCoord(NamedTuple):
    x: float
    y: float


class Dimensions(NamedTuple):
    width: int
    height: int


PixelDimensions = Dimensions
Pixels = int
Meters = float
Canvas2D = np.ndarray  # of shape (y, x)

MAP_BOUNDARY_MARGIN: Meters = 300


class MapBoundaries(NamedTuple):
    """Distances in carla.World coordinates"""

    min_x: Meters
    min_y: Meters
    max_x: Meters
    max_y: Meters


class CroppingRect(NamedTuple):
    x: int
    y: int
    width: int
    height: int

    @property
    def vslice(self) -> slice:
        return slice(self.y, self.y + self.height)

    @property
    def hslice(self) -> slice:
        return slice(self.x, self.x + self.width)


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


class RenderingWindow(NamedTuple):
    origin: carla.Location
    area: PixelDimensions


class MapMaskGenerator:
    """Generates 2D, top-down representations of a map.

    Each mask covers area specified by rendering window or whole map (when rendering window is disabled).
    Note that layer, mask, canvas are somewhat interchangeable terms for the same thing.

    Rendering is implemented using OpenCV, so it can be easily adjusted
    to become a regular RGB renderer (just change all `color` arguments to 3-element tuples)
    """

    def __init__(self, client, pixels_per_meter: int) -> None:
        self.client = client
        self.pixels_per_meter = pixels_per_meter
        self.rendering_window: Optional[RenderingWindow] = None

        self._world = client.get_world()
        self._map = self._world.get_map()
        self._topology = self._map.get_topology()
        self._waypoints = self._map.generate_waypoints(2)
        self._map_boundaries = self._find_map_boundaries()
        self._each_road_waypoints = self._generate_road_waypoints()
        self._mask_size: PixelDimensions = self.calculate_mask_size()

    def _find_map_boundaries(self) -> MapBoundaries:
        """Find extreme locations on a map.

        It adds a decent margin because waypoints lie on the road, which means
        that anything that is slightly further than the boundary
        could cause out-of-range exceptions (e.g. pavements, walkers, etc.)
        """
        return MapBoundaries(
            min_x=min(
                self._waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            - MAP_BOUNDARY_MARGIN,
            min_y=min(
                self._waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            - MAP_BOUNDARY_MARGIN,
            max_x=max(
                self._waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            + MAP_BOUNDARY_MARGIN,
            max_y=max(
                self._waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            + MAP_BOUNDARY_MARGIN,
        )

    def calculate_mask_size(self) -> PixelDimensions:
        """Convert map boundaries to pixel resolution."""
        width_in_meters = self._map_boundaries.max_x - self._map_boundaries.min_x
        height_in_meters = self._map_boundaries.max_y - self._map_boundaries.min_y
        width_in_pixels = int(width_in_meters * self.pixels_per_meter)
        height_in_pixels = int(height_in_meters * self.pixels_per_meter)
        return PixelDimensions(width=width_in_pixels, height=height_in_pixels)

    def disable_local_rendering_mode(self):
        self.rendering_window = None

    def enable_local_rendering_mode(self, rendering_window: RenderingWindow):
        self.rendering_window = rendering_window

    def location_to_pixel(self, loc: carla.Location) -> Coord:
        """Convert world coordinates to pixel coordinates.

        For example: top leftmost location will be a pixel at (0, 0).
        """
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y

        # Pixel coordinates on full map
        x = int(self.pixels_per_meter * (loc.x - min_x))
        y = int(self.pixels_per_meter * (loc.y - min_y))

        if self.rendering_window is not None:
            # global rendering area coordinates
            origin_x = self.pixels_per_meter * (self.rendering_window.origin.x - min_x)
            origin_y = self.pixels_per_meter * (self.rendering_window.origin.y - min_y)
            topleft_x = int(origin_x - self.rendering_window.area.width / 2)
            topleft_y = int(origin_y - self.rendering_window.area.height / 2)

            # x, y becomes local coordinates within rendering window
            x -= topleft_x
            y -= topleft_y

        return Coord(x=int(x), y=int(y))

    def make_empty_mask(self) -> Mask:
        if self.rendering_window is None:
            shape = (self._mask_size.height, self._mask_size.width)
        else:
            shape = (
                self.rendering_window.area.height,
                self.rendering_window.area.width,
            )
        return np.zeros(shape, np.uint8)

    def _generate_road_waypoints(self) -> List[RoadSegmentWaypoints]:
        """Return all, precisely located waypoints from the map.

        Topology contains simplified representation (a start and an end
        waypoint for each road segment). By expanding each until another
        road segment is found, we explore all possible waypoints on the map.

        Returns a list of waypoints for each road segment.
        """
        precision: Meters = 0.05
        road_segments_starts: carla.Waypoint = [
            road_start for road_start, road_end in self._topology
        ]

        each_road_waypoints = []
        for road_start_waypoint in road_segments_starts:
            road_waypoints = [road_start_waypoint]

            # Generate as long as it's the same road
            next_waypoints = road_start_waypoint.next(precision)

            if len(next_waypoints) > 0:
                # Always take first (may be at intersection)
                next_waypoint = next_waypoints[0]
                while next_waypoint.road_id == road_start_waypoint.road_id:
                    road_waypoints.append(next_waypoint)
                    next_waypoint = next_waypoint.next(precision)

                    if len(next_waypoint) > 0:
                        next_waypoint = next_waypoint[0]
                    else:
                        # Reached the end of road segment
                        break
            each_road_waypoints.append(road_waypoints)
        return each_road_waypoints

    def road_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        # FIXME Refactor that crap
        for road_waypoints in self._each_road_waypoints:
            road_left_side = [
                lateral_shift(w.transform, -w.lane_width * 0.5) for w in road_waypoints
            ]
            road_right_side = [
                lateral_shift(w.transform, w.lane_width * 0.5) for w in road_waypoints
            ]

            polygon = road_left_side + [x for x in reversed(road_right_side)]
            polygon = [self.location_to_pixel(x) for x in polygon]

            # Create road objects here

            if len(polygon) > 2:
                polygon = np.array([polygon], dtype=np.int32)

                # FIXME Hard to notice the difference without polylines
                cv.polylines(
                    img=canvas, pts=polygon, isClosed=True, color=COLOR_ON, thickness=5
                )
                cv.fillPoly(img=canvas, pts=polygon, color=COLOR_ON)
        return canvas

    def lanes_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        for road_waypoints in self._each_road_waypoints:
            # if not road_waypoints[0].is_junction:
            # NOTE This block was inside if statement - some junctions may not have proper lane markings drawn
            # Left Side
            lanes.draw_lane_marking_single_side(
                canvas,
                road_waypoints,
                side=LaneSide.LEFT,
                location_to_pixel_func=self.location_to_pixel,
                color=COLOR_ON,
            )

            # Right Side
            lanes.draw_lane_marking_single_side(
                canvas,
                road_waypoints,
                side=LaneSide.RIGHT,
                location_to_pixel_func=self.location_to_pixel,
                color=COLOR_ON,
            )
        return canvas

    def centerlines_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        for road_waypoints in self._each_road_waypoints:
            polygon = [
                self.location_to_pixel(wp.transform.location) for wp in road_waypoints
            ]
            if len(polygon) > 2:
                polygon = np.array([polygon], dtype=np.int32)
                cv.polylines(
                    img=canvas, pts=polygon, isClosed=False, color=COLOR_ON, thickness=1
                )
        return canvas

    def agent_vehicle_mask(self, agent: carla.Actor) -> Mask:
        canvas = self.make_empty_mask()
        bb = agent.bounding_box.extent
        corners = [
            carla.Location(x=-bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=bb.y),
        ]

        agent.get_transform().transform(corners)
        corners = [self.location_to_pixel(loc) for loc in corners]
        cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        return canvas

    def vehicles_mask(self, vehicles: List[carla.Actor]) -> Mask:
        canvas = self.make_empty_mask()
        for veh in vehicles:
            bb = veh.bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
            ]

            veh.get_transform().transform(corners)
            corners = [self.location_to_pixel(loc) for loc in corners]

            if veh.attributes["role_name"] == "hero":
                color = COLOR_OFF
            else:
                color = COLOR_ON

            cv.fillPoly(img=canvas, pts=np.int32([corners]), color=color)
        return canvas

    def pedestrians_mask(self, pedestrians: List[carla.Actor]) -> Mask:
        canvas = self.make_empty_mask()
        for ped in pedestrians:
            if not hasattr(ped, "bounding_box"):
                continue

            bb = ped.bounding_box.extent
            corners = [
                carla.Location(x=-2 * bb.x, y=-2 * bb.y),
                carla.Location(x=2 * bb.x, y=-2 * bb.y),
                carla.Location(x=2 * bb.x, y=2 * bb.y),
                carla.Location(x=-2 * bb.x, y=2 * bb.y),
            ]

            ped.get_transform().transform(corners)
            corners = [self.location_to_pixel(loc) for loc in corners]
            cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        return canvas

    def traffic_lights_masks(self, traffic_lights: List[carla.Actor]) -> Tuple[Mask]:
        red_light_canvas = self.make_empty_mask()
        yellow_light_canvas = self.make_empty_mask()
        green_light_canvas = self.make_empty_mask()
        tls = carla.TrafficLightState
        for tl in traffic_lights:
            world_pos = tl.get_location()
            pos = self.location_to_pixel(world_pos)
            radius = int(self.pixels_per_meter * 1.2)
            if tl.state == tls.Red:
                target_canvas = red_light_canvas
            elif tl.state == tls.Yellow:
                target_canvas = yellow_light_canvas
            elif tl.state == tls.Green:
                target_canvas = green_light_canvas
            else:
                # Unknown or off traffic light
                continue

            cv.circle(
                img=target_canvas,
                center=pos,
                radius=radius,
                color=COLOR_ON,
                thickness=cv.FILLED,
            )
        return red_light_canvas, yellow_light_canvas, green_light_canvas


class MapMaskGeneratorObjectLevel:
    """Generates 2D, top-down representations of a map.

    Each mask covers area specified by rendering window or whole map (when rendering window is disabled).
    Note that layer, mask, canvas are somewhat interchangeable terms for the same thing.

    Rendering is implemented using OpenCV, so it can be easily adjusted
    to become a regular RGB renderer (just change all `color` arguments to 3-element tuples)
    """

    def __init__(self, client, radius) -> None:
        self.client = client
        self.radius = radius
        self.rendering_window: Optional[RenderingWindow] = None

        self._world = client.get_world()
        self._map = self._world.get_map()
        self._topology = self._map.get_topology()
        self._waypoints = self._map.generate_waypoints(2)
        self._map_boundaries = self._find_map_boundaries()
        self._each_road_waypoints = self._generate_road_waypoints()
        self._mask_size: PixelDimensions = self.calculate_mask_size()

    def _find_map_boundaries(self) -> MapBoundaries:
        """Find extreme locations on a map.

        It adds a decent margin because waypoints lie on the road, which means
        that anything that is slightly further than the boundary
        could cause out-of-range exceptions (e.g. pavements, walkers, etc.)
        """
        return MapBoundaries(
            min_x=min(
                self._waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            - MAP_BOUNDARY_MARGIN,
            min_y=min(
                self._waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            - MAP_BOUNDARY_MARGIN,
            max_x=max(
                self._waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            + MAP_BOUNDARY_MARGIN,
            max_y=max(
                self._waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            + MAP_BOUNDARY_MARGIN,
        )

    def calculate_mask_size(self) -> PixelDimensions:
        """Convert map boundaries to pixel resolution."""
        width_in_meters = self._map_boundaries.max_x - self._map_boundaries.min_x
        height_in_meters = self._map_boundaries.max_y - self._map_boundaries.min_y
        width_in_pixels = int(width_in_meters)
        height_in_pixels = int(height_in_meters)
        return PixelDimensions(width=width_in_pixels, height=height_in_pixels)

    def disable_local_rendering_mode(self):
        self.rendering_window = None

    def enable_local_rendering_mode(self, rendering_window: RenderingWindow):
        self.rendering_window = rendering_window

    def location_to_pixel(self, loc: carla.Location) -> carla.Location:
        """Convert world coordinates to pixel coordinates.

        For example: top leftmost location will be a pixel at (0, 0).
        """
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y

        # Pixel coordinates on full map
        x = loc.x - min_x
        y = loc.y - min_y

        if self.rendering_window is not None:
            # global rendering area coordinates
            origin_x = self.rendering_window.origin.x - min_x
            origin_y = self.rendering_window.origin.y - min_y
            topleft_x = origin_x - self.rendering_window.area.width / 2
            topleft_y = origin_y - self.rendering_window.area.height / 2

            # x, y becomes local coordinates within rendering window
            x -= topleft_x
            y -= topleft_y

        return FltCoord(x=x, y=y)

    def make_empty_mask(self) -> Mask:
        if self.rendering_window is None:
            shape = (self._mask_size.height, self._mask_size.width)
        else:
            shape = (
                self.rendering_window.area.height,
                self.rendering_window.area.width,
            )
        return np.zeros(shape, np.uint8)

    def _generate_road_waypoints(self) -> List[RoadSegmentWaypoints]:
        """Return all, precisely located waypoints from the map.

        Topology contains simplified representation (a start and an end
        waypoint for each road segment). By expanding each until another
        road segment is found, we explore all possible waypoints on the map.

        Returns a list of waypoints for each road segment.
        """
        precision: Meters = 0.05
        road_segments_starts: carla.Waypoint = [
            road_start for road_start, road_end in self._topology
        ]

        each_road_waypoints = []
        for road_start_waypoint in road_segments_starts:
            road_waypoints = [road_start_waypoint]

            # Generate as long as it's the same road
            next_waypoints = road_start_waypoint.next(precision)

            if len(next_waypoints) > 0:
                # Always take first (may be at intersection)
                next_waypoint = next_waypoints[0]
                while next_waypoint.road_id == road_start_waypoint.road_id:
                    road_waypoints.append(next_waypoint)
                    next_waypoint = next_waypoint.next(precision)

                    if len(next_waypoint) > 0:
                        next_waypoint = next_waypoint[0]
                    else:
                        # Reached the end of road segment
                        break
            each_road_waypoints.append(road_waypoints)
        return each_road_waypoints

    def road_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        # FIXME Refactor that crap
        for road_waypoints in self._each_road_waypoints:
            road_left_side = [
                lateral_shift(w.transform, -w.lane_width * 0.5) for w in road_waypoints
            ]
            road_right_side = [
                lateral_shift(w.transform, w.lane_width * 0.5) for w in road_waypoints
            ]

            polygon = road_left_side + [x for x in reversed(road_right_side)]
            polygon = [self.location_to_pixel(x) for x in polygon]

            # Create road objects here

            if len(polygon) > 2:
                polygon = np.array([polygon], dtype=np.int32)

                # FIXME Hard to notice the difference without polylines
                cv.polylines(
                    img=canvas, pts=polygon, isClosed=True, color=COLOR_ON, thickness=5
                )
                cv.fillPoly(img=canvas, pts=polygon, color=COLOR_ON)
        return canvas

    def lanes_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        for road_waypoints in self._each_road_waypoints:
            # if not road_waypoints[0].is_junction:
            # NOTE This block was inside if statement - some junctions may not have proper lane markings drawn
            # Left Side
            lanes.draw_lane_marking_single_side(
                canvas,
                road_waypoints,
                side=LaneSide.LEFT,
                location_to_pixel_func=self.location_to_pixel,
                color=COLOR_ON,
            )

            # Right Side
            lanes.draw_lane_marking_single_side(
                canvas,
                road_waypoints,
                side=LaneSide.RIGHT,
                location_to_pixel_func=self.location_to_pixel,
                color=COLOR_ON,
            )
        return canvas

    def centerlines_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        for road_waypoints in self._each_road_waypoints:
            polygon = [
                self.location_to_pixel(wp.transform.location) for wp in road_waypoints
            ]
            if len(polygon) > 2:
                polygon = np.array([polygon], dtype=np.int32)
                cv.polylines(
                    img=canvas, pts=polygon, isClosed=False, color=COLOR_ON, thickness=1
                )
        return canvas

    def all_objects(
        self,
        agent: carla.Actor,
        vehicles: List[carla.Actor],
        pedestrians: List[carla.Actor],
        traffic_lights: List[carla.Actor],
    ):
        # print("all_objects are", agent, vehicles, pedestrians, traffic_lights)
        objects = []

        ego_rotation = agent.get_transform().rotation
        ego_matrix = np.array(agent.get_transform().get_matrix())

        ego_extent = agent.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.0
        ego_yaw = ego_rotation.yaw / 180 * np.pi

        # also add ego box for visulization
        relative_yaw = 0
        relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

        # add vehicle velocity and brake flag
        ego_transform = agent.get_transform()
        ego_control = agent.get_control()
        ego_velocity = agent.get_velocity()
        ego_speed = self._get_forward_speed(
            transform=ego_transform, velocity=ego_velocity
        )  # In m/s
        ego_brake = ego_control.brake

        result = {
            "class": "Vehicle",
            "extent": [
                ego_dx[2],
                ego_dx[0],
                ego_dx[1],
            ],  # NOTE: height stored in first dimension
            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
            "yaw": relative_yaw,
            "num_points": -1,
            "distance": -1,
            "speed": ego_speed,
            "brake": ego_brake,
            "id": int(agent.id),
            "ego_matrix": agent.get_transform().get_matrix(),
            "ego_vehicle": True,
        }

        objects.append(result)

        # for vehicle in vehicles:
        for vehicle in vehicles:
            if vehicle.get_location().distance(agent.get_location()) < self.radius:
                if vehicle.id != agent.id:
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())
                    vehicle_id = vehicle.id

                    vehicle_extent = vehicle.bounding_box.extent
                    dx = (
                        np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z])
                        * 2.0
                    )
                    yaw = vehicle_rotation.yaw / 180 * np.pi

                    relative_yaw = yaw - ego_yaw
                    relative_pos = self.get_relative_transform(
                        ego_matrix, vehicle_matrix
                    )

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control = vehicle.get_control()
                    vehicle_velocity = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(
                        transform=vehicle_transform, velocity=vehicle_velocity
                    )  # In m/s
                    vehicle_brake = vehicle_control.brake

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Vehicle",
                        "extent": [
                            dx[2],
                            dx[0],
                            dx[1],
                        ],  # NOTE: height stored in first dimension
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "distance": distance,
                        "speed": vehicle_speed,
                        "brake": vehicle_brake,
                        "id": int(vehicle_id),
                        "ego_matrix": vehicle.get_transform().get_matrix(),
                        "ego_vehicle": False,
                    }
                    objects.append(result)

        for pedestrian in pedestrians:
            if pedestrian.get_location().distance(agent.get_location()) < self.radius:
                if pedestrian.id != agent.id:
                    pedestrian_rotation = pedestrian.get_transform().rotation
                    pedestrian_matrix = np.array(
                        pedestrian.get_transform().get_matrix()
                    )
                    pedestrian_id = pedestrian.id

                    pedestrian_extent = pedestrian.bounding_box.extent
                    dx = (
                        np.array(
                            [
                                pedestrian_extent.x,
                                pedestrian_extent.y,
                                pedestrian_extent.z,
                            ]
                        )
                        * 2.0
                    )
                    yaw = pedestrian_rotation.yaw / 180 * np.pi

                    relative_yaw = yaw - ego_yaw
                    relative_pos = self.get_relative_transform(
                        ego_matrix, pedestrian_matrix
                    )

                    pedestrian_transform = pedestrian.get_transform()
                    pedestrian_velocity = pedestrian.get_velocity()
                    pedestrian_speed = self._get_forward_speed(
                        transform=pedestrian_transform, velocity=pedestrian_velocity
                    )  # In m/s

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Pedestrian",
                        "extent": [
                            dx[2],
                            dx[0],
                            dx[1],
                        ],  # NOTE: height stored in first dimension
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "distance": distance,
                        "speed": pedestrian_speed,
                        "id": int(pedestrian_id),
                        "ego_matrix": pedestrian.get_transform().get_matrix(),
                    }
                    objects.append(result)

        for traffic_light in traffic_lights:
            traffic_light_distance = traffic_light.get_location().distance(
                agent.get_location()
            )

            traffic_light_matrix = np.array(traffic_light.get_transform().get_matrix())

            relative_pos = self.get_2d_relative_transform(
                ego_matrix, traffic_light_matrix
            )

            traffic_light_pos = traffic_light.get_transform().location

            distance = np.linalg.norm(relative_pos)

            if traffic_light.state == carla.TrafficLightState.Red:
                state = "red"
            elif traffic_light.state == carla.TrafficLightState.Yellow:
                state = "yellow"
            elif traffic_light.state == carla.TrafficLightState.Green:
                state = "green"
            else:
                # Unknown or off traffic light
                continue

            distance = np.linalg.norm(relative_pos)

            if traffic_light_distance < self.radius:
                traffic_light_matrix = np.array(
                    traffic_light.get_transform().get_matrix()
                )
                traffic_light_id = traffic_light.id

                traffic_light_extent = traffic_light.bounding_box.extent
                dx = (
                    np.array(
                        [
                            traffic_light_extent.x,
                            traffic_light_extent.y,
                            traffic_light_extent.z,
                        ]
                    )
                    * 2.0
                )

                relative_pos = self.get_relative_transform(
                    ego_matrix, traffic_light_matrix
                )

                if traffic_light.state == carla.TrafficLightState.Red:
                    state = "red"
                elif traffic_light.state == carla.TrafficLightState.Yellow:
                    state = "yellow"
                elif traffic_light.state == carla.TrafficLightState.Green:
                    state = "green"
                else:
                    # Unknown or off traffic light
                    continue

                distance = np.linalg.norm(relative_pos)

                result = {
                    "class": "TrafficLight",
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "distance": distance,
                    "id": int(traffic_light_id),
                    "state": state,
                    "ego_matrix": traffic_light.get_transform().get_matrix(),
                }

                objects.append(result)
            else:
                # print(
                #     # "Traffic light is too far away.",
                #     "distance:",
                #     traffic_light_distance,
                #     "Odistance:",
                #     distance,
                #     "ego_matrix:",
                #     ego_matrix[:3, 3],
                #     "tl_matrix:",
                #     traffic_light_matrix[:3, 3],
                #     "Pos:",
                #     relative_pos,
                #     "TL position:",
                #     traffic_light_pos,
                #     "ID:",
                #     traffic_light.id,
                # )
                pass
        return objects

    # def agent_vehicle_mask(self, agent: carla.Actor) -> Mask:
    #     canvas = self.make_empty_mask()
    #     bb = agent.bounding_box.extent
    #     corners = [
    #         carla.Location(x=-bb.x, y=-bb.y),
    #         carla.Location(x=bb.x, y=-bb.y),
    #         carla.Location(x=bb.x, y=bb.y),
    #         carla.Location(x=-bb.x, y=bb.y),
    #     ]

    #     agent.get_transform().transform(corners)
    #     corners = [self.location_to_pixel(loc) for loc in corners]

    #     # "class": "Car",
    #     # "extent": [
    #     #     ego_dx[2],
    #     #     ego_dx[0],
    #     #     ego_dx[1],
    #     # ],  # NOTE: height stored in first dimension
    #     # "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
    #     # "yaw": relative_yaw,
    #     # "num_points": -1,
    #     # "distance": -1,
    #     # "speed": ego_speed,
    #     # "brake": ego_brake,
    #     # "id": int(agent.id),
    #     # "ego_matrix": agent.get_transform().get_matrix(),

    #     start_dict = {}

    #     # cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
    #     return canvas

    # def vehicles_mask(self, vehicles: List[carla.Actor]) -> Mask:
    #     canvas = self.make_empty_mask()
    #     for veh in vehicles:
    #         bb = veh.bounding_box.extent
    #         corners = [
    #             carla.Location(x=-bb.x, y=-bb.y),
    #             carla.Location(x=bb.x, y=-bb.y),
    #             carla.Location(x=bb.x, y=bb.y),
    #             carla.Location(x=-bb.x, y=bb.y),
    #         ]

    #         veh.get_transform().transform(corners)
    #         corners = [self.location_to_pixel(loc) for loc in corners]

    #         if veh.attributes["role_name"] == "hero":
    #             color = COLOR_OFF
    #         else:
    #             color = COLOR_ON

    #         cv.fillPoly(img=canvas, pts=np.int32([corners]), color=color)
    #     return canvas

    # def agent_vehicle_mask(self, agent: carla.Actor) -> Mask:
    #     canvas = self.make_empty_mask()
    #     bb = agent.bounding_box.extent
    #     corners = [
    #         carla.Location(x=-bb.x, y=-bb.y),
    #         carla.Location(x=bb.x, y=-bb.y),
    #         carla.Location(x=bb.x, y=bb.y),
    #         carla.Location(x=-bb.x, y=bb.y),
    #     ]

    #     agent.get_transform().transform(corners)
    #     corners = [self.location_to_pixel(loc) for loc in corners]

    #     # "class": "Car",
    #     # "extent": [
    #     #     ego_dx[2],
    #     #     ego_dx[0],
    #     #     ego_dx[1],
    #     # ],  # NOTE: height stored in first dimension
    #     # "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
    #     # "yaw": relative_yaw,
    #     # "num_points": -1,
    #     # "distance": -1,
    #     # "speed": ego_speed,
    #     # "brake": ego_brake,
    #     # "id": int(agent.id),
    #     # "ego_matrix": agent.get_transform().get_matrix(),

    #     start_dict = {}

    #     # cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
    #     return canvas

    # def vehicles_mask(self, vehicles: List[carla.Actor]) -> Mask:
    #     canvas = self.make_empty_mask()
    #     for veh in vehicles:
    #         bb = veh.bounding_box.extent
    #         corners = [
    #             carla.Location(x=-bb.x, y=-bb.y),
    #             carla.Location(x=bb.x, y=-bb.y),
    #             carla.Location(x=bb.x, y=bb.y),
    #             carla.Location(x=-bb.x, y=bb.y),
    #         ]

    #         veh.get_transform().transform(corners)
    #         corners = [self.location_to_pixel(loc) for loc in corners]

    #         if veh.attributes["role_name"] == "hero":
    #             color = COLOR_OFF
    #         else:
    #             color = COLOR_ON

    #         cv.fillPoly(img=canvas, pts=np.int32([corners]), color=color)
    #     return canvas

    def pedestrians_mask(self, pedestrians: List[carla.Actor]) -> Mask:
        canvas = self.make_empty_mask()
        for ped in pedestrians:
            if not hasattr(ped, "bounding_box"):
                continue

            bb = ped.bounding_box.extent
            corners = [
                carla.Location(x=-2 * bb.x, y=-2 * bb.y),
                carla.Location(x=2 * bb.x, y=-2 * bb.y),
                carla.Location(x=2 * bb.x, y=2 * bb.y),
                carla.Location(x=-2 * bb.x, y=2 * bb.y),
            ]

            ped.get_transform().transform(corners)
            corners = [self.location_to_pixel(loc) for loc in corners]
            cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        return canvas

    def traffic_lights_masks(self, traffic_lights: List[carla.Actor]) -> Tuple[Mask]:
        tls = carla.TrafficLightState

        objects = []

        for tl in traffic_lights:
            world_pos = tl.get_location()
            pos = self.location_to_pixel(world_pos)
            if tl.state == tls.Red:
                target_canvas = "red"
            elif tl.state == tls.Yellow:
                target_canvas = "yellow"
            elif tl.state == tls.Green:
                target_canvas = "green"
            else:
                # Unknown or off traffic light
                continue

            objects.append(
                {
                    "type": "traffic_light",
                    "position": [pos.x, pos.y],
                    "target_canvas": target_canvas,
                }
            )

        return objects

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos

        # transform to right handed system
        relative_pos[1] = -relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        # trans = -np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos  # + trans

        return relative_pos

    def get_2d_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos

        # transform to right handed system
        relative_pos[1] = -relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        # trans = -np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos  # + trans

        relative_pos = relative_pos[:2]

        return relative_pos

    def _get_forward_speed(self, transform=None, velocity=None):
        """Convert the vehicle transform directly to forward speed"""
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        speed = np.dot(vel_np, orientation)
        return speed
