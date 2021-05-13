import numpy as np

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def blobby_mask(h,w,max_percentage=0.25):

    percentage_covered = 0
    mask = np.zeros((h,w))
    center = np.round(np.random.rand(2) * np.array([h-1,w-1]))
    while percentage_covered < max_percentage:
        percentage_missing = max_percentage - percentage_covered
        radius = np.random.rand() * np.sqrt(percentage_missing * h * w / np.pi) # ensure that the radius is not too large
        new_blob = create_circular_mask(h,w,center=center,radius=radius)
        mask = np.logical_or(mask, new_blob)
        percentage_covered = percentage_covered + np.sum(mask)/(h*w)
        pos = np.where(mask)
        idx = np.random.choice(len(pos)-1)
        center = [pos[1][idx],pos[0][idx]]

        # allow breaking before covering the maximum percentage:
        if np.random.rand()/max_percentage * percentage_covered > 1/2:
            break

    return mask


if __name__=="__main__":
    h,w = 28,28
    m = blobby_mask(h,w,1/4)
    #print(m)
    print("percentage covered:", np.sum(m)/h/w)