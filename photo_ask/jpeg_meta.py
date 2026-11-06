"""사진을 외부로 보내기 전에 픽셀만 남긴 새 JPEG로 다시 저장한다.

사진 앱 미리보기에는 좌표·찍은 시각·기기 모델(EXIF), XMP, Photoshop 구간, 파일 끝에 붙은 보조 이미지(MPF)가
그대로 있다. 구간을 골라 지우면 형식마다 빠져나갈 길이 남으므로 원본 바이트를 넘기지 않는다.
회전 값은 픽셀에 적용한다. 다시 저장하므로 화질이 조금 줄고, 색 프로필이 빠져 넓은 색역 사진은 색이 조금 달라질 수 있다.
"""
import io

from PIL import Image, ImageOps

JPEG_QUALITY = 90


class JpegMetadataError(ValueError):
    """사진을 다시 저장하지 못했다. 호출한 쪽은 이 파일을 보내면 안 된다."""


def strip_metadata(data: bytes) -> bytes:
    """JPEG 바이트를 디코딩해 픽셀만 담은 새 JPEG 바이트를 돌려준다."""
    try:
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as source:
            pixels = ImageOps.exif_transpose(source).convert("RGB")
        clean = Image.new("RGB", pixels.size)   # 저장할 때 원본 info(주석 등)가 따라가지 않게 빈 이미지에 옮긴다
        clean.paste(pixels)
        out = io.BytesIO()
        clean.save(out, format="JPEG", quality=JPEG_QUALITY)
    except Exception as e:  # noqa: BLE001 — 어떤 이유로든 실패하면 보내지 않는다
        raise JpegMetadataError(f"{type(e).__name__}: {str(e)[:200]}") from e
    return out.getvalue()
