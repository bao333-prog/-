from __future__ import annotations

import io
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


PROFILES = {"baseline", "standard", "targeted", "gentle"}
RESIZE_MODES = {"center_crop", "letterbox"}


def prepare_image(image: Image.Image, *, image_size: int, resize_mode: str) -> Image.Image:
    """统一输入处理；补边模式保留整张主体，兼容旧实验的中心裁剪模式。"""
    if resize_mode not in RESIZE_MODES:
        raise ValueError("unsupported resize_mode: " + resize_mode)
    if resize_mode == "center_crop":
        return ImageOps.fit(image.convert("RGB"), (image_size, image_size),
                            method=Image.Resampling.BILINEAR)
    image = ImageOps.exif_transpose(image).convert("RGBA")
    image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    return ImageOps.pad(image, (image_size, image_size),
                        method=Image.Resampling.BILINEAR, color="white")


def degrade_resolution(image: Image.Image, *, scale: float) -> Image.Image:
    if not 0 < scale <= 1:
        raise ValueError("scale must be in the interval (0, 1]")
    if scale == 1:
        return image.copy()
    reduced = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(reduced, Image.Resampling.BILINEAR).resize(
        image.size, Image.Resampling.BILINEAR
    )


def adjust_color_temperature(image: Image.Image, factor: float) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    pixels[..., 0] *= factor
    pixels[..., 2] /= max(factor, 1e-6)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    table = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
    return image.point(table * 3)


def add_sensor_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    numpy_rng = np.random.default_rng(rng.randrange(2**32))
    noise = numpy_rng.normal(0, sigma, size=pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8), mode="RGB")


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def apply_steam(image: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    numpy_rng = np.random.default_rng(rng.randrange(2**32))
    mask_values = numpy_rng.integers(0, 256, size=(image.height, image.width), dtype=np.uint8)
    mask = Image.fromarray(mask_values, mode="L").filter(
        ImageFilter.GaussianBlur(radius=max(2, min(image.size) / 8))
    )
    mask = ImageEnhance.Contrast(mask).enhance(0.35)
    mask = mask.point(lambda value: round(value * strength))
    return Image.composite(Image.new("RGB", image.size, "white"), image, mask)


def apply_glare(image: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = rng.randint(max(2, image.width // 5), max(3, image.width // 2))
    height = rng.randint(max(2, image.height // 8), max(3, image.height // 3))
    left = rng.randint(-width // 2, max(-width // 2, image.width - width // 2))
    top = rng.randint(-height // 2, max(-height // 2, image.height - height // 2))
    alpha = round(255 * max(0, min(strength, 1)))
    draw.ellipse((left, top, left + width, top + height), fill=(255, 245, 220, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def apply_occlusion(image: Image.Image, coverage: float, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(1, round(image.width * min(max(coverage, 0), 0.5)))
    height = max(1, round(image.height * min(max(coverage, 0), 0.5)))
    left = rng.randint(0, max(0, image.width - width))
    top = rng.randint(0, max(0, image.height - height))
    color = rng.choice([(35, 25, 15, 110), (210, 195, 160, 90)])
    draw.ellipse((left, top, left + width, top + height), fill=color)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


@dataclass
class ProxyAugmenter:
    profile: str
    image_size: int = 224
    seed: int | None = None
    resize_mode: str = "center_crop"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(
                f"augmentation profile must be one of {', '.join(sorted(PROFILES))}"
            )
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError("unsupported resize_mode: " + self.resize_mode)
        if self.profile == "gentle" and self.resize_mode != "letterbox":
            raise ValueError("gentle augmentation requires resize_mode='letterbox'")
        self._rng = random.Random(self.seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.profile == "baseline":
            return prepare_image(image, image_size=self.image_size, resize_mode=self.resize_mode)

        if self.profile == "gentle":
            # 轻量增强不随机裁剪主体；小角度旋转先扩展画布，随后缩放补边。
            image = prepare_image(image, image_size=self.image_size, resize_mode=self.resize_mode)
            if self._rng.random() < 0.5:
                image = ImageOps.mirror(image)
            image = image.rotate(
                self._rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR,
                expand=True, fillcolor="white",
            )
            image = ImageEnhance.Brightness(image).enhance(self._rng.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(self._rng.uniform(0.9, 1.1))
            image = ImageEnhance.Color(image).enhance(self._rng.uniform(0.9, 1.1))
            return prepare_image(image, image_size=self.image_size, resize_mode="letterbox")

        # 组合使用旧增强与补边时，也先统一方向和透明背景；旧center_crop行为保持不变。
        image = (prepare_image(image, image_size=self.image_size, resize_mode=self.resize_mode)
                 if self.resize_mode == "letterbox" else image.convert("RGB"))
        image = self._random_crop(image)
        if self._rng.random() < 0.5:
            image = ImageOps.mirror(image)
        image = image.rotate(
            self._rng.uniform(-6, 6),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        image = ImageEnhance.Brightness(image).enhance(self._rng.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(self._rng.uniform(0.8, 1.2))
        image = ImageEnhance.Color(image).enhance(self._rng.uniform(0.75, 1.25))
        image = ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
        )

        if self.profile == "targeted":
            image = adjust_color_temperature(image, self._rng.uniform(0.72, 1.35))
            image = adjust_gamma(image, self._rng.uniform(0.75, 1.35))
            if self._rng.random() < 0.65:
                image = degrade_resolution(image, scale=self._rng.uniform(0.35, 0.8))
            if self._rng.random() < 0.55:
                image = image.filter(ImageFilter.GaussianBlur(self._rng.uniform(0.3, 1.5)))
            if self._rng.random() < 0.55:
                image = add_sensor_noise(image, self._rng.uniform(2, 10), self._rng)
            if self._rng.random() < 0.5:
                image = jpeg_roundtrip(image, self._rng.randint(35, 80))
            if self._rng.random() < 0.5:
                image = apply_steam(image, self._rng.uniform(0.08, 0.28), self._rng)
            if self._rng.random() < 0.4:
                image = apply_glare(image, self._rng.uniform(0.15, 0.45), self._rng)
            if self._rng.random() < 0.3:
                image = apply_occlusion(image, self._rng.uniform(0.08, 0.25), self._rng)
        return image.convert("RGB")

    def _random_crop(self, image: Image.Image) -> Image.Image:
        side = min(image.size)
        crop_side = max(1, round(side * self._rng.uniform(0.78, 1.0)))
        left = self._rng.randint(0, max(0, image.width - crop_side))
        top = self._rng.randint(0, max(0, image.height - crop_side))
        return image.crop((left, top, left + crop_side, top + crop_side))

