#!/usr/bin/env python3
"""Generates ICON0.PNG, the 144x80 thumbnail the PSP shows in the XMB.

Written with only zlib and struct so the build has no image-library
dependency. The 5x7 bitmap font below is drawn at integer scales, which
keeps every edge crisp at the exact size the PSP displays.
"""

import struct
import sys
import zlib

WIDTH, HEIGHT = 144, 80

BACKGROUND = (0x0A, 0x0E, 0x12)
BORDER = (0x22, 0x2C, 0x38)
TITLE = (0x6E, 0xE7, 0x87)
RULE = (0x1F, 0x6F, 0x3F)
SUBTITLE = (0x8B, 0x94, 0x9E)

FONT = {
    'A': ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    'B': ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    'C': ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    'D': ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    'E': ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    'F': ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    'G': ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    'H': ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    'I': ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    'L': ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    'M': ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    'N': ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    'O': ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    'P': ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    'R': ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    'S': ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    'T': ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    'U': ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    '0': ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    '9': ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    '-': ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ' ': ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def blank():
    return [[BACKGROUND] * WIDTH for _ in range(HEIGHT)]


def put_pixel(image, x, y, colour):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        image[y][x] = colour


def rectangle(image, x, y, width, height, colour):
    for row in range(y, y + height):
        for column in range(x, x + width):
            put_pixel(image, column, row, colour)


def text_width(text, scale, gap):
    if not text:
        return 0
    return len(text) * (5 * scale + gap) - gap


def draw_text(image, text, x, y, scale, gap, colour):
    for character in text:
        glyph = FONT.get(character)
        if glyph is None:
            raise KeyError("no glyph for %r" % character)
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    rectangle(image, x + column * scale, y + row * scale,
                              scale, scale, colour)
        x += 5 * scale + gap


def write_png(path, image):
    raw = bytearray()
    for row in image:
        raw.append(0)                      # filter type 0 for every scanline
        for red, green, blue in row:
            raw += bytes((red, green, blue))

    def chunk(tag, payload):
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body +
                struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", header))
        handle.write(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        handle.write(chunk(b"IEND", b""))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ICON0.PNG"
    image = blank()

    # Thin frame so the icon reads as a tile against the XMB wallpaper.
    rectangle(image, 0, 0, WIDTH, 1, BORDER)
    rectangle(image, 0, HEIGHT - 1, WIDTH, 1, BORDER)
    rectangle(image, 0, 0, 1, HEIGHT, BORDER)
    rectangle(image, WIDTH - 1, 0, 1, HEIGHT, BORDER)

    title, title_scale, title_gap = "LLMPSP", 3, 3
    title_x = (WIDTH - text_width(title, title_scale, title_gap)) // 2
    draw_text(image, title, title_x, 16, title_scale, title_gap, TITLE)

    rectangle(image, title_x, 44, text_width(title, title_scale, title_gap),
              2, RULE)

    subtitle, subtitle_scale, subtitle_gap = "FALCON 90M", 2, 2
    subtitle_x = (WIDTH - text_width(subtitle, subtitle_scale,
                                     subtitle_gap)) // 2
    draw_text(image, subtitle, subtitle_x, 54, subtitle_scale, subtitle_gap,
              SUBTITLE)

    write_png(path, image)
    print("wrote %s (%dx%d)" % (path, WIDTH, HEIGHT))


if __name__ == "__main__":
    main()
