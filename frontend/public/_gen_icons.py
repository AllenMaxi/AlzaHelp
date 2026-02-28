#!/usr/bin/env python3
"""Generate PWA icon PNGs for AlzaHelp using pure Python (no dependencies)."""
import struct, zlib, os

BASE = os.path.dirname(os.path.abspath(__file__))
BG = (124, 58, 237)   # #7c3aed
FG = (255, 255, 255)  # white

# Simple bitmap font for "AH" - each char is defined on a grid
# We'll define A and H as pixel patterns (1=foreground, 0=background)
# Each glyph is 5 wide x 7 tall
GLYPHS = {
    'A': [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
    ],
    'H': [
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
    ],
}

def render_text(text, size):
    """Render text onto a size x size pixel grid. Returns list of rows, each row is list of (r,g,b)."""
    glyph_w, glyph_h = 5, 7
    spacing = 1
    total_chars = len(text)
    text_w = total_chars * glyph_w + (total_chars - 1) * spacing  # 11 for "AH"
    text_h = glyph_h  # 7

    # Scale factor: make text about 1/3 of icon size
    target_h = size // 3
    scale = max(1, target_h // glyph_h)

    scaled_w = text_w * scale
    scaled_h = text_h * scale

    # Offsets to center
    ox = (size - scaled_w) // 2
    oy = (size - scaled_h) // 2

    # Build the glyph map for entire text
    text_pixels = []
    for row_i in range(glyph_h):
        row = []
        for ci, ch in enumerate(text):
            g = GLYPHS.get(ch, GLYPHS['A'])
            row.extend(g[row_i])
            if ci < total_chars - 1:
                row.extend([0] * spacing)
        text_pixels.append(row)

    # Build image
    pixels = []
    for y in range(size):
        row = []
        for x in range(size):
            # Check if in text region
            tx = x - ox
            ty = y - oy
            if 0 <= tx < scaled_w and 0 <= ty < scaled_h:
                gx = tx // scale
                gy = ty // scale
                if gx < len(text_pixels[0]) and gy < len(text_pixels) and text_pixels[gy][gx]:
                    # Add slight rounding: skip corners of each scaled pixel block for smoother look
                    row.append(FG)
                    continue
            # Background - add subtle radial gradient for polish
            cx, cy = size / 2, size / 2
            dist = ((x - cx)**2 + (y - cy)**2) ** 0.5
            max_dist = (cx**2 + cy**2) ** 0.5
            factor = 1.0 - 0.15 * (dist / max_dist)
            r = max(0, min(255, int(BG[0] * factor)))
            g = max(0, min(255, int(BG[1] * factor)))
            b = max(0, min(255, int(BG[2] * factor)))
            row.append((r, g, b))
        pixels.append(row)

    # Add rounded corners (transparent-ish corners replaced with slightly darker)
    corner_r = size // 8
    for y in range(size):
        for x in range(size):
            # Check four corners
            for cy, cx in [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]:
                dx = abs(x - cx)
                dy = abs(y - cy)
                if dx < corner_r and dy < corner_r:
                    if dx + dy > corner_r:
                        pass  # keep as is
                    # Actual rounded corner check
                    ccx = corner_r if cx == 0 else size - 1 - corner_r
                    ccy = corner_r if cy == 0 else size - 1 - corner_r
                    if ((x - ccx)**2 + (y - ccy)**2) > corner_r**2:
                        if (cx == 0 and x < corner_r) or (cx == size-1 and x > size-1-corner_r):
                            if (cy == 0 and y < corner_r) or (cy == size-1 and y > size-1-corner_r):
                                pixels[y][x] = (0, 0, 0)  # black corner (will be masked by OS)

    return pixels


def make_png(pixels, size):
    """Create a PNG file from pixel data. Returns bytes."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'

    # IHDR
    ihdr = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)  # 8-bit RGB

    # IDAT
    raw = b''
    for row in pixels:
        raw += b'\x00'  # filter: none
        for r, g, b in row:
            raw += struct.pack('BBB', r, g, b)

    compressed = zlib.compress(raw)

    # IEND
    return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')


for size in [192, 512]:
    pixels = render_text("AH", size)
    data = make_png(pixels, size)
    path = os.path.join(BASE, f"icon-{size}.png")
    with open(path, 'wb') as f:
        f.write(data)
    print(f"Created {path} ({len(data)} bytes)")
