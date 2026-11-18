import struct
import sys
import _webp

data = open(sys.argv[1], "rb").read()
decoder = _webp.WebPAnimDecoder(data)
(width, height), *_ = decoder.get_info()
pixels = decoder.get_next()[0]
bgra = bytearray(pixels)
bgra[0::4], bgra[2::4] = pixels[2::4], pixels[0::4]
header = struct.pack("<2sIHHI", b"BM", 54 + len(bgra), 0, 0, 54)
header += struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, len(bgra), 0, 0, 0, 0)
with open(sys.argv[2], "wb") as output:
    output.write(header + bgra)

