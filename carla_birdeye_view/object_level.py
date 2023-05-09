import carla
import logging
import numpy as np

from enum import IntEnum, auto, Enum
from pathlib import Path
from typing import List
from filelock import FileLock

from carla_birdeye_view import actors, cache
from carla_birdeye_view.actors import SegregatedActors
from carla_birdeye_view.colors import RGB
from carla_birdeye_view.mask import (
    PixelDimensions,
    Coord,
    CroppingRect,
    MapMaskGeneratorObjectLevel,
    Mask,
    COLOR_ON,
    RenderingWindow,
    Dimensions,
    MAP_BOUNDARY_MARGIN,
)
from carla_birdeye_view import (
    BirdView,
    RgbCanvas,
    RGB_BY_MASK,
    BIRDVIEW_SHAPE_CHW,
    BIRDVIEW_SHAPE_HWC,
    rotate,
    circle_circumscribed_around_rectangle,
    square_fitting_rect_at_any_rotation,
)
import carla_birdeye_view

# __all__ = ["BirdViewProducer", "DEFAULT_HEIGHT", "DEFAULT_WIDTH"]

import cv2.cv2 as cv2

# Np print options set to 2 decimal places and suppress scientific notation
np.set_printoptions(precision=2, suppress=True)

import random


def rotate(image, angle, center=None, scale=1.0):
    assert image.dtype == np.uint8

    """Copy paste of imutils method but with INTER_NEAREST and BORDER_CONSTANT flags"""
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # return the rotated image
    return rotated


def circle_circumscribed_around_rectangle(rect_size: Dimensions) -> float:
    """Returns radius of that circle."""
    a = rect_size.width / 2
    b = rect_size.height / 2
    return float(np.sqrt(np.power(a, 2) + np.power(b, 2)))


def square_fitting_rect_at_any_rotation(rect_size: Dimensions) -> float:
    """Preview: https://pasteboard.co/J1XK62H.png"""
    radius = circle_circumscribed_around_rectangle(rect_size)
    side_length_of_square_circumscribed_around_circle = radius * 2
    return side_length_of_square_circumscribed_around_circle


class BirdViewProducerObjectLevel:
    """Responsible for producing top-down view on the map, following agent's vehicle.

    About BirdView:
    - top-down view, fixed directly above the agent (including vehicle rotation), cropped to desired size
    - consists of stacked layers (masks), each filled with ones and zeros (depends on MaskMaskGenerator implementation).
        Example layers: road, vehicles, pedestrians. 0 indicates -> no presence in that pixel, 1 -> presence
    - convertible to RGB image
    - Rendering full road and lanes masks is computationally expensive, hence caching mechanism is used
    """

    def __init__(
        self,
        client: carla.Client,
        target_size: PixelDimensions,
        pixels_per_meter: int = 4,
        crop_type: carla_birdeye_view.BirdViewCropType = carla_birdeye_view.BirdViewCropType.FRONT_AND_REAR_AREA,
    ) -> None:
        self.client = client
        self.target_size = target_size
        self._pixels_per_meter = pixels_per_meter
        # Divide target size by pixels_per_meter to get size in meters
        # self.target_size =
        # Calculate the radius of the circle circumscribed around the target_size rectangle
        self.radius = circle_circumscribed_around_rectangle(
            PixelDimensions(
                width=target_size.width / pixels_per_meter,
                height=target_size.height / pixels_per_meter,
            )
        )

        # print("Radius is " + str(self.radius))

        self._crop_type = crop_type

        # print("Crop type is " + str(crop_type))

        if crop_type is carla_birdeye_view.BirdViewCropType.FRONT_AND_REAR_AREA:
            rendering_square_size = round(
                square_fitting_rect_at_any_rotation(self.target_size)
            )

            self.origin = Coord(
                x=rendering_square_size // 2, y=rendering_square_size // 2
            )
        elif crop_type is carla_birdeye_view.BirdViewCropType.FRONT_AREA_ONLY:
            # We must keep rendering size from FRONT_AND_REAR_AREA (in order to avoid rotation issues)
            enlarged_size = PixelDimensions(
                width=target_size.width, height=target_size.height * 2
            )
            rendering_square_size = round(
                square_fitting_rect_at_any_rotation(enlarged_size)
            )
            self.origin = Coord(x=rendering_square_size // 2, y=0)
        else:
            raise NotImplementedError
        self.rendering_area = PixelDimensions(
            width=rendering_square_size, height=rendering_square_size
        )

        # Rotate by 90 degrees then translate by origin
        self.global_rotation_translation = np.array(
            [
                [0, -1, self.origin.x],
                [1, 0, self.origin.y],
                [0, 0, 1],
            ]
        )

        self._world = client.get_world()
        self._map = self._world.get_map()
        self.masks_generator = MapMaskGeneratorObjectLevel(client, radius=self.radius)

        # cache_path = self.parametrized_cache_path()
        # if Path(cache_path).is_file():
        #     carla_birdeye_view.LOGGER.info(f"Loading cache from {cache_path}")
        #     with FileLock(f"{cache_path}.lock"):
        #         static_cache = np.load(cache_path)
        #         self.full_road_cache = static_cache[0]
        #         self.full_lanes_cache = static_cache[1]
        #         self.full_centerlines_cache = static_cache[2]
        #     carla_birdeye_view.LOGGER.info(
        #         f"Loaded static layers from cache file: {cache_path}"
        #     )
        # else:
        #     carla_birdeye_view.LOGGER.warning(
        #         f"Cache file does not exist, generating cache at {cache_path}"
        #     )
        #     self.full_road_cache = self.masks_generator.road_mask()
        #     self.full_lanes_cache = self.masks_generator.lanes_mask()
        #     self.full_centerlines_cache = self.masks_generator.centerlines_mask()
        #     static_cache = np.stack(
        #         [
        #             self.full_road_cache,
        #             self.full_lanes_cache,
        #             self.full_centerlines_cache,
        #         ]
        #     )
        #     with FileLock(f"{cache_path}.lock"):
        #         np.save(cache_path, static_cache, allow_pickle=False)
        #     carla_birdeye_view.LOGGER.info(
        #         f"Saved static layers to cache file: {cache_path}"
        #     )

    def parametrized_cache_path(self) -> str:
        cache_dir = Path("birdview_v2_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        opendrive_content_hash = cache.generate_opendrive_content_hash(self._map)
        cache_filename = (
            f"{self._map.name}__"
            f"px_per_meter={self._pixels_per_meter}__"
            f"opendrive_hash={opendrive_content_hash}__"
            f"margin={MAP_BOUNDARY_MARGIN}.npy"
        )
        return str(cache_dir / cache_filename)

    def produce(
        self,
        agent_vehicle: carla.Actor,
    ) -> BirdView:
        all_actors = actors.query_all(world=self._world)
        segregated_actors = actors.segregate_by_type(actors=all_actors)
        agent_vehicle_loc = agent_vehicle.get_location()

        # Reusing already generated static masks for whole map
        self.masks_generator.disable_local_rendering_mode()
        agent_global_px_pos = self.masks_generator.location_to_pixel(agent_vehicle_loc)

        # cropping_rect = CroppingRect(
        #     x=int(agent_global_px_pos.x - self.rendering_area.width / 2),
        #     y=int(agent_global_px_pos.y - self.rendering_area.height / 2),
        #     width=self.rendering_area.width,
        #     height=self.rendering_area.height,
        # )

        objects = []

        # masks = np.zeros(
        #     shape=(
        #         len(BirdViewMasks),
        #         self.rendering_area.height,
        #         self.rendering_area.width,
        #     ),
        #     dtype=np.uint8,
        # )

        # masks[BirdViewMasks.ROAD.value] = self.full_road_cache[
        #     cropping_rect.vslice, cropping_rect.hslice
        # ]
        # masks[BirdViewMasks.LANES.value] = self.full_lanes_cache[
        #     cropping_rect.vslice, cropping_rect.hslice
        # ]

        # masks[BirdViewMasks.CENTERLINES.value] = self.full_centerlines_cache[
        #     cropping_rect.vslice, cropping_rect.hslice
        # ]

        # Dynamic masks

        # rendering_window = RenderingWindow(
        #     origin=agent_vehicle_loc, area=self.rendering_area
        # )

        # self.masks_generator.enable_local_rendering_mode(rendering_window)
        # masks, objects = self._render_actors_masks(
        #     agent_vehicle, segregated_actors, masks
        # )

        objects = self.masks_generator.all_objects(
            agent_vehicle,
            segregated_actors.vehicles,
            segregated_actors.pedestrians,
            segregated_actors.traffic_lights,
        )
        # cropped_masks = self.apply_agent_following_transformation_to_masks(
        #     agent_vehicle,
        #     masks,
        # )
        # ordered_indices = [
        #     mask.value for mask in carla_birdeye_view.BirdViewMasks.bottom_to_top()
        # ]
        return objects

    # @staticmethod
    def as_rgb(self, objects) -> RgbCanvas:
        """Converts objects to RGB canvas."""

        canvas = np.zeros(
            shape=(self.target_size.width, self.target_size.height, 3), dtype=np.uint8
        )

        for object in objects:
            self.render(canvas, object)

        # _, h, w = birdview.shape
        # rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        # nonzero_indices = lambda arr: arr == COLOR_ON

        # for mask_type in carla_birdeye_view.BirdViewMasks.bottom_to_top():
        #     rgb_color = RGB_BY_MASK[mask_type]
        #     mask = birdview[mask_type]
        #     # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
        #     rgb_canvas[nonzero_indices(mask)] = rgb_color
        # return rgb_canvas
        return canvas

    def render(self, canvas: RgbCanvas, object) -> None:
        """Renders object on canvas."""
        if object["class"] == "TrafficLight":
            # print("TrafficLight detected")
            self.render_light(canvas, object)
            return

        if object["class"] == "Vehicle":
            if object["ego_vehicle"]:
                color = carla_birdeye_view.RGB_BY_MASK[
                    carla_birdeye_view.BirdViewMasks.AGENT
                ]
            else:
                color = carla_birdeye_view.RGB_BY_MASK[
                    carla_birdeye_view.BirdViewMasks.VEHICLES
                ]
        elif object["class"] == "Pedestrian":
            color = carla_birdeye_view.RGB_BY_MASK[
                carla_birdeye_view.BirdViewMasks.PEDESTRIANS
            ]
        else:
            raise ValueError(f"Unknown object class: {object['class']}")

        self.render_polygon(
            canvas,
            object["extent"][1],
            object["extent"][2],
            object["position"][0],
            object["position"][1],
            object["yaw"],
            color,
        )

    def render_light(self, canvas: RgbCanvas, object) -> None:
        """Renders traffic light on canvas."""
        corners = np.asarray(
            [
                [
                    object["extent"][0] * self._pixels_per_meter,
                    object["extent"][1] * self._pixels_per_meter,
                    1,
                ]
            ]
        )

        corners = np.dot(self.global_rotation_translation, corners.T).T

        # print("Rotated corners: ", corners)
        # Remove third dimension (z)
        corners = corners[:, :2]

        if object["state"] == "red":
            color = carla_birdeye_view.RGB_BY_MASK[
                carla_birdeye_view.BirdViewMasks.RED_LIGHTS
            ]
        elif object["state"] == "yellow":
            color = carla_birdeye_view.RGB_BY_MASK[
                carla_birdeye_view.BirdViewMasks.YELLOW_LIGHTS
            ]
        elif object["state"] == "green":
            color = carla_birdeye_view.RGB_BY_MASK[
                carla_birdeye_view.BirdViewMasks.GREEN_LIGHTS
            ]
        else:
            raise ValueError(
                "Unknown traffic light state encountered: ", object["state"]
            )

        # Render as a circle with radius 1.5m
        cv2.circle(
            canvas,
            center=corners.astype(np.int32),
            radius=int(1.5 * self._pixels_per_meter),
            color=color,
            thickness=cv2.FILLED,
        )

    def render_polygon(
        self, canvas: RgbCanvas, width, height, x, y, angle, color
    ) -> None:
        """Renders polygon on canvas."""
        corners = self.get_corners(width, height, x, y, angle)
        corners = corners.astype(np.int32)
        # print(corners)
        cv2.fillPoly(canvas, pts=np.int32([corners]), color=color)

    def get_corners(self, width, height, x, y, angle):
        """Returns corners of polygon.
        Corners are first calculated in local coordinate system (i.e. with respect to center of polygon).
        Then, we create a rotation translation matrix to move the polygon to its correct position.
        """
        corners = np.array(
            [
                [-width * self._pixels_per_meter / 2, -height / 2, 1],
                [width * self._pixels_per_meter / 2, -height / 2, 1],
                [width * self._pixels_per_meter / 2, height / 2, 1],
                [-width * self._pixels_per_meter / 2, height / 2, 1],
            ]
        )
        # print("Initial corners: ", corners)
        # Scale the polygon
        # corners *= self._pixels_per_meter

        # print("Scaled corners: ", corners)

        rotation_translation_matrix = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                    x * self._pixels_per_meter,
                ],
                [
                    np.sin(angle),
                    np.cos(angle),
                    y * self._pixels_per_meter,
                ],
                [0, 0, 1],
            ]
        )

        corners = np.dot(rotation_translation_matrix, corners.T).T

        # Apply global rotation and translation
        corners = np.dot(self.global_rotation_translation, corners.T).T

        # print("Rotated corners: ", corners)
        # Remove third dimension (z)
        corners = corners[:, :2]

        # print("Final corners: ", corners)
        return corners

    def _render_actors_masks(
        self,
        agent_vehicle: carla.Actor,
        segregated_actors: SegregatedActors,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Fill masks with ones and zeros (more precisely called as "bitmask").
        Although numpy dtype is still the same, additional semantic meaning is being added.
        """
        # light_objects = self.masks_generator.traffic_lights_masks(
        #     segregated_actors.traffic_lights
        # )

        # vehicle_objects = self.masks_generator.vehicles_masks(
        #     agent_vehicle, segregated_actors.vehicles
        # )

        # masks[BirdViewMasks.AGENT.value] = self.masks_generator.agent_vehicle_mask(
        #     agent_vehicle
        # )
        # masks[BirdViewMasks.VEHICLES.value] = self.masks_generator.vehicles_mask(
        #     segregated_actors.vehicles
        # )
        # masks[BirdViewMasks.PEDESTRIANS.value] = self.masks_generator.pedestrians_mask(
        #     segregated_actors.pedestrians
        # )
        return masks

    def apply_agent_following_transformation_to_masks(
        self,
        agent_vehicle: carla.Actor,
        masks: np.ndarray,
    ) -> np.ndarray:
        agent_transform = agent_vehicle.get_transform()
        angle = (
            agent_transform.rotation.yaw + 90
        )  # vehicle's front will point to the top

        # Rotating around the center
        crop_with_car_in_the_center = masks
        masks_n, h, w = crop_with_car_in_the_center.shape
        rotation_center = Coord(x=w // 2, y=h // 2)

        # warpAffine from OpenCV requires the first two dimensions to be in order: height, width, channels
        crop_with_centered_car = np.transpose(
            crop_with_car_in_the_center, axes=(1, 2, 0)
        )
        rotated = rotate(crop_with_centered_car, angle, center=rotation_center)
        rotated = np.transpose(rotated, axes=(2, 0, 1))

        half_width = self.target_size.width // 2
        hslice = slice(rotation_center.x - half_width, rotation_center.x + half_width)

        if self._crop_type is carla_birdeye_view.BirdViewCropType.FRONT_AREA_ONLY:
            vslice = slice(
                rotation_center.y - self.target_size.height, rotation_center.y
            )
        elif self._crop_type is carla_birdeye_view.BirdViewCropType.FRONT_AND_REAR_AREA:
            half_height = self.target_size.height // 2
            vslice = slice(
                rotation_center.y - half_height, rotation_center.y + half_height
            )
        else:
            raise NotImplementedError
        assert (
            vslice.start > 0 and hslice.start > 0
        ), "Trying to access negative indexes is not allowed, check for calculation errors!"
        car_on_the_bottom = rotated[:, vslice, hslice]
        return car_on_the_bottom


class BirdViewProducerObjectLevelRenderer:
    def __init__(
        self,
        target_size: PixelDimensions,
        pixels_per_meter: int = 4,
        crop_type: carla_birdeye_view.BirdViewCropType = carla_birdeye_view.BirdViewCropType.FRONT_AND_REAR_AREA,
        num_slots: int = 10,
    ):
        self.target_size = target_size
        self._pixels_per_meter = pixels_per_meter
        self._crop_type = crop_type
        self._num_slots = num_slots

        # Divide target size by pixels_per_meter to get size in meters
        # self.target_size =
        # Calculate the radius of the circle circumscribed around the target_size rectangle
        self.radius = circle_circumscribed_around_rectangle(
            PixelDimensions(
                width=target_size.width / pixels_per_meter,
                height=target_size.height / pixels_per_meter,
            )
        )

        # print("Radius is " + str(self.radius))
        self._crop_type = crop_type

        # print("Crop type is " + str(crop_type))

        if crop_type is carla_birdeye_view.BirdViewCropType.FRONT_AND_REAR_AREA:
            # rendering_square_size = round(
            #     square_fitting_rect_at_any_rotation(self.target_size)
            # )

            self.origin = Coord(
                x=self.target_size.width // 2, y=self.target_size.height // 2
            )
        elif crop_type is carla_birdeye_view.BirdViewCropType.FRONT_AREA_ONLY:
            # We must keep rendering size from FRONT_AND_REAR_AREA (in order to avoid rotation issues)
            # enlarged_size = PixelDimensions(
            #     width=target_size.width, height=target_size.height * 2
            # )
            # rendering_square_size = round(
            #     square_fitting_rect_at_any_rotation(enlarged_size)
            # )
            self.origin = Coord(
                x=self.target_size.width // 2, y=self.target_size.height
            )
        else:
            raise NotImplementedError
        # self.rendering_area = PixelDimensions(
        #     width=rendering_square_size, height=rendering_square_size
        # )

        # Rotate by 90 degrees then translate by origin
        self.global_rotation_translation = np.array(
            [
                [0, 1, self.origin.x],
                [-1, 0, self.origin.y],
                [0, 0, 1],
            ]
        )

    def as_rgb(self, objects):
        """Converts a list of objects to RGB image."""
        canvas = np.zeros(
            shape=(self.target_size.width, self.target_size.height, 3), dtype=np.uint8
        )

        for object in objects:
            self.render(canvas, object)

        return canvas

    def render(self, canvas: RgbCanvas, object) -> None:
        """Renders object on canvas."""
        print("Object: ", object)
        if object["class"] == "TrafficLight":
            self.render_light(canvas, object)
            return

        if object["class"] == "Vehicle":
            if object["ego_vehicle"]:
                color = carla_birdeye_view.RGB_BY_MASK[
                    carla_birdeye_view.BirdViewMasks.AGENT
                ]
            else:
                color = carla_birdeye_view.RGB_BY_MASK[
                    carla_birdeye_view.BirdViewMasks.VEHICLES
                ]
        elif object["class"] == "Pedestrian":
            color = carla_birdeye_view.RGB_BY_MASK[
                carla_birdeye_view.BirdViewMasks.PEDESTRIANS
            ]
        else:
            raise ValueError(f"Unknown object class: {object['class']}")

        self.render_polygon(
            canvas,
            object["extent"][1],
            object["extent"][2],
            object["position"][0],
            -object["position"][1],
            object["yaw"],
            color,
        )

    def render_light(self, canvas: RgbCanvas, object, forced_color=None) -> None:
        """Renders traffic light on canvas."""
        corners = np.asarray(
            [
                [
                    object["position"][0] * self._pixels_per_meter,
                    -object["position"][1] * self._pixels_per_meter,
                    1,
                ]
            ]
        )

        corners = np.dot(self.global_rotation_translation, corners.T).T

        # print("Rotated corners: ", corners)
        # Remove third dimension (z)
        corners = corners[0, :2].astype(np.int32)

        if forced_color is not None:
            color = forced_color
        else:
            if object["state"] == "red":
                color = carla_birdeye_view.RGB_BY_MASK[
                    carla_birdeye_view.BirdViewMasks.RED_LIGHTS
                ]
            elif object["state"] == "yellow":
                color = carla_birdeye_view.RGB_BY_MASK[
                    carla_birdeye_view.BirdViewMasks.YELLOW_LIGHTS
                ]
            elif object["state"] == "green":
                color = carla_birdeye_view.RGB_BY_MASK[
                    carla_birdeye_view.BirdViewMasks.GREEN_LIGHTS
                ]
            else:
                raise ValueError(
                    "Unknown traffic light state encountered: ", object["state"]
                )
        print(
            "rendering light of color", color, "at location", corners.astype(np.int32)
        )

        # Render as a circle with radius 1.5m
        # print("rendering light of color", color, "at location", location)
        cv2.circle(
            canvas,
            center=(corners[0], corners[1]),
            radius=int(1.5 * self._pixels_per_meter),
            color=color,
            thickness=cv2.FILLED,
        )

    def render_polygon(
        self, canvas: RgbCanvas, width, height, x, y, angle, color
    ) -> None:
        """Renders polygon on canvas."""
        corners = self.get_corners(width, height, x, y, angle)
        corners = corners.astype(np.int32)
        # print(corners)
        cv2.fillPoly(canvas, pts=np.int32([corners]), color=color)

    def get_corners(self, width, height, x, y, angle):
        """Returns corners of polygon.
        Corners are first calculated in local coordinate system (i.e. with respect to center of polygon).
        Then, we create a rotation translation matrix to move the polygon to its correct position.
        """
        corners = np.array(
            [
                [
                    -width * self._pixels_per_meter / 2,
                    -height * self._pixels_per_meter / 2,
                    1,
                ],
                [
                    width * self._pixels_per_meter / 2,
                    -height * self._pixels_per_meter / 2,
                    1,
                ],
                [
                    width * self._pixels_per_meter / 2,
                    height * self._pixels_per_meter / 2,
                    1,
                ],
                [
                    -width * self._pixels_per_meter / 2,
                    height * self._pixels_per_meter / 2,
                    1,
                ],
            ]
        )
        # print("Initial corners: ", corners)
        # Scale the polygon
        # corners *= self._pixels_per_meter

        # print("Scaled corners: ", corners)

        rotation_translation_matrix = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                    x * self._pixels_per_meter,
                ],
                [
                    np.sin(angle),
                    np.cos(angle),
                    y * self._pixels_per_meter,
                ],
                [0, 0, 1],
            ]
        )

        corners = np.dot(rotation_translation_matrix, corners.T).T

        # Apply global rotation and translation
        corners = np.dot(self.global_rotation_translation, corners.T).T

        # print("Rotated corners: ", corners)
        # Remove third dimension (z)
        corners = corners[:, :2]

        # print("Final corners: ", corners)
        return corners

    def as_slots(self, objects, randomize=True):
        """Converts a list of objects to slots."""
        canvas = [
            np.zeros(
                shape=(
                    self.target_size.width,
                    self.target_size.height,
                    1,
                ),
                dtype=np.uint8,
            )
            for _ in range(self._num_slots)
        ]
        # Remove ego vehicle
        objects = [
            object
            for object in objects
            if not ("ego_vehicle" in object and object["ego_vehicle"])
        ]

        if randomize:
            objects = random.sample(objects, len(objects))

        # Single object per slot, last slot contains all remaining objects
        slot_tuples = [
            (min(self._num_slots - 1, i), object) for i, object in enumerate(objects)
        ]

        for slot_idx, object in slot_tuples:
            self.render_slot(object, canvas[slot_idx])
            print("RENDERED")

        canvas = np.concatenate(canvas, axis=2)

        return canvas

    def render_slot(self, object, slot_canvas):
        """Renders object on slot canvas."""
        if object["class"] in ["Vehicle", "Pedestrian"]:
            self.render_polygon(
                slot_canvas,
                object["extent"][1],
                object["extent"][2],
                object["position"][0],
                -object["position"][1],
                object["yaw"],
                (255,),
            )
        elif object["class"] == "TrafficLight":
            self.render_light(slot_canvas, object, forced_color=(255,))
        else:
            raise ValueError(f"Unknown object class: {object['class']}")

    def point_in_bounds(self, x, y):
        point_local = np.array([x, y, 1])

        # Apply global rotation and translation
        point_global = np.dot(self.global_rotation_translation, point_local)

        x, y = point_global[:2]

        return self._in_bounds(x, y)

    def corners_in_bounds(self, corners):
        return any([self._in_bounds(x, y) for x, y in corners])

    def _in_bounds(self, x, y):
        """Checks if x, y is in bounds of the canvas."""
        return 0 <= x < self.target_size.width and 0 <= y < self.target_size.height

    def filter_objects_in_scene(self, objects, exclude_ego_vehicle=True):
        """Filters objects in scene to only include those that are relevant for the bird's eye view."""
        filtered_objects = []
        for object in objects:
            if object["class"] in ["Vehicle", "Pedestrian"]:
                if (
                    exclude_ego_vehicle
                    and "ego_vehicle" in object
                    and object["ego_vehicle"]
                ):
                    continue
                corners = self.get_corners(
                    object["extent"][1],
                    object["extent"][2],
                    object["position"][0],
                    -object["position"][1],
                    object["yaw"],
                )
                if self.corners_in_bounds(corners):
                    filtered_objects.append(object)

            elif object["class"] == "TrafficLight":
                if self.point_in_bounds(
                    object["position"][0] * self._pixels_per_meter,
                    -object["position"][1] * self._pixels_per_meter,
                ):
                    filtered_objects.append(object)
        return filtered_objects
