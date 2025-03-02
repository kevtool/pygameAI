import numpy as np

def argmax(logits):
    return np.argmax(logits)

def argsort(seq, reverse=False):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

# rect is a pygame rect, radius is circle radius, center is circle center
def intersects(rect, radius, center) -> bool:
    circle_distance_x = abs(center[0]-rect.centerx)
    circle_distance_y = abs(center[1]-rect.centery)
    if circle_distance_x > rect.w/2.0+radius or circle_distance_y > rect.h/2.0+radius:
        return False
    if circle_distance_x <= rect.w/2.0 or circle_distance_y <= rect.h/2.0:
        return True
    corner_x = circle_distance_x-rect.w/2.0
    corner_y = circle_distance_y-rect.h/2.0
    corner_distance_sq = corner_x**2.0 +corner_y**2.0
    return corner_distance_sq <= radius**2.0