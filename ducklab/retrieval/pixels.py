"""Select a newly appeared object from a probability map, independent of training libraries."""

import numpy as np


def component(prob, threshold=0.5, min_pixels=4):
    """Pick the highest integrated confidence 8-connected foreground component."""
    pixels = set(zip(*np.nonzero(prob >= threshold)))
    best = None
    best_score = 0
    while pixels:
        todo = [pixels.pop()]
        region = []
        while todo:
            y, x = todo.pop()
            region.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    near = (y + dy, x + dx)
                    if near in pixels:
                        pixels.remove(near)
                        todo.append(near)
        if len(region) >= min_pixels:
            yy, xx = np.array(region).T
            score = prob[yy, xx].sum()
            if score > best_score:
                best_score = score
                best = (yy, xx)
    return best
