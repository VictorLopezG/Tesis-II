import numpy as np
import cv2


def normalize(image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil, radpixels, angulardiv):
	"""
	Description:
		Normalize iris region by unwraping the circular region into a rectangular
		block of constant dimensions.

	Input:
		image		- Input iris image.

		x_iris		- x-coordinate of the circle defining the iris boundary.
		y_iris		- y-coordinate of the circle defining the iris boundary.
		r_iris		- Radius of the circle defining the iris boundary.

		x_pupil		- x-coordinate of the circle defining the pupil boundary.
		y_pupil		- y-coordinate of the circle defining the pupil boundary.
		r_pupil		- Radius of the circle defining the pupil boundary.

		radpixels	- Radial resolution (vertical dimension).
		angulardiv	- Angular resolution (horizontal dimension).

	Output:
		polar_array	- Normalized form of the iris region.
		polar_noise	- Normalized form of the noise region.
	"""
	radiuspixels = radpixels + 2
	angledivisions = angulardiv-1

	r = np.arange(radiuspixels)
	# crea el arreglo inicial para la textura del iris (360 grados en un arreglo del 0 al 2pi)
	theta = np.linspace(0, 2*np.pi, angledivisions+1)

	# Calculate displacement of pupil center from the iris center
	ox = x_pupil - x_iris
	oy = y_pupil - y_iris

	if ox <= 0:
		sgn = -1
	elif ox > 0:
		sgn = 1

	if ox == 0 and oy > 0:
		sgn = 1

	a = np.ones(angledivisions+1) * (ox**2 + oy**2)

	# Need to do something for ox = 0
	if ox == 0:
		phi = np.pi/2
	else:
		phi = np.arctan(oy/ox)

	b = sgn * np.cos(np.pi - phi - theta)

	# Calculate radius around the iris as a function of the angle
	r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - r_iris**2))
	r = np.array([r - r_pupil])

	rmat = np.dot(np.ones([radiuspixels, 1]), r)

	rmat = rmat * np.dot(np.ones([angledivisions+1, 1]), np.array([np.linspace(0, 1, radiuspixels)])).transpose()
	rmat = rmat + r_pupil

	# Exclude values at the boundary of the pupil iris border, and the iris scelra border
	# as these may not correspond to areas in the iris region and will introduce noise.
	# ie don't take the outside rings as iris data.
	rmat = rmat[1:radiuspixels-1, :]

	# Calculate cartesian location of each data point around the circular iris region
	xcosmat = np.dot(np.ones([radiuspixels-2, 1]), np.array([np.cos(theta)]))
	xsinmat = np.dot(np.ones([radiuspixels-2, 1]), np.array([np.sin(theta)]))

	xo = rmat * xcosmat
	yo = rmat * xsinmat

	xo = x_pupil + xo
	xo = np.round(xo).astype(int)
	coords = np.where(xo >= image.shape[1])
	xo[coords] = image.shape[1] - 1
	coords = np.where(xo < 0)
	xo[coords] = 0
	
	yo = y_pupil - yo
	yo = np.round(yo).astype(int)
	coords = np.where(yo >= image.shape[0])
	yo[coords] = image.shape[0] - 1
	coords = np.where(yo < 0)
	yo[coords] = 0
	
	# Extract intensity values into the normalised polar representation through interpolation
	polar_array = image[yo, xo]
	polar_array = polar_array / 255
	#cv2.imshow("im",polar_array)
	# Create noise array with location of NaNs in polar_array
	polar_noise = np.zeros(polar_array.shape)
	coords = np.where(np.isnan(polar_array))
	polar_noise[coords] = 1
	
	# Get rid of outling points in order to write out the circular pattern
	image[yo, xo] = 255

	# Get pixel coords for circle around iris
	x, y = circlecoords([x_iris, y_iris], r_iris, image.shape)
	image[y, x] = 255

	# Get pixel coords for circle around pupil
	xp, yp = circlecoords([x_pupil, y_pupil], r_pupil, image.shape)
	image[yp, xp] = 255
	
	# Replace NaNs before performing feature encoding
	coords = np.where((np.isnan(polar_array)))
	
	polar_array2 = polar_array
	
	polar_array2[coords] = 0.5
	
	avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
	polar_array[coords] = avg

	return polar_array, polar_noise.astype(bool)


def circlecoords(c, r, imgsize, nsides=600):
	"""
	Description:
		Find the coordinates of a circle based on its centre and radius.

	Input:
		c   	- Centre of the circle.
		r  		- Radius of the circle.
		imgsize - Size of the image that the circle will be plotted onto.
		nsides 	- Number of sides of the convex-hull bodering the circle (default as 600).

	Output:
		x,y     - Circle coordinates.
	"""
	a = np.linspace(0, 2*np.pi, 2*nsides+1)
	xd = np.round(r * np.cos(a) + c[0])
	yd = np.round(r * np.sin(a) + c[1])

	#  Get rid of values larger than image
	xd2 = xd
	coords = np.where(xd >= imgsize[1])
	xd2[coords[0]] = imgsize[1] - 1
	coords = np.where(xd < 0)
	xd2[coords[0]] = 0

	yd2 = yd
	coords = np.where(yd >= imgsize[0])
	yd2[coords[0]] = imgsize[0] - 1
	coords = np.where(yd < 0)
	yd2[coords[0]] = 0

	x = np.round(xd2).astype(int)
	y = np.round(yd2).astype(int)
	return x, y


import cv2
import numpy as np

def normalize_iris(image, pupil_center, pupil_radius, iris_center, iris_radius, radial_res=64, angular_res=512):
    """
    Normalizes a circular iris region into a rectangular block.
    """
    # Create an empty template for the unrolled iris rectangle
    normalized_iris = np.zeros((radial_res, angular_res))
    
    # Calculate angles from 0 to 2*pi
    thetas = np.linspace(0, 2 * np.pi, angular_res)
    
    for j, theta in enumerate(thetas):
        # Locate boundary points on pupil and iris edges at specific angle theta
        xp = pupil_center[0] + pupil_radius * np.cos(theta)
        yp = pupil_center[1] + pupil_radius * np.sin(theta)
        
        xi = iris_center[0] + iris_radius * np.cos(theta)
        yi = iris_center[1] + iris_radius * np.sin(theta)
        
        # Linearly sample points between the pupil border and iris border
        for i in range(radial_res):
            r = i / radial_res
            # Reconstruct cartesian coordinates for target pixels
            x = int((1 - r) * xp + r * xi)
            y = int((1 - r) * yp + r * yi)
            
            # Map pixel intensity if it falls inside image dimensions
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                normalized_iris[i, j] = image[y, x]
                
    return np.uint8(normalized_iris)

import numpy as np
import cv2

def rubber_sheet_inverse(polar_img, center, r_min, r_max):
    """
    Transforms a polar/normalized iris image back to Cartesian coordinates.
    
    Parameters:
    - polar_img: NumPy array of shape (radius_size, theta_size), the unwrapped image.
    - center: Tuple (cx, cy) of the pupil/iris center in the original image.
    - r_min: Inner radius (pupil boundary).
    - r_max: Outer radius (iris boundary).
    
    Returns:
    - cart_img: Reconstructed Cartesian image containing the un-normalized region.
    """
    num_radii, num_thetas = polar_img.shape
    cx, cy = center
    
    # Estimate size of output Cartesian canvas
    max_radius_int = int(np.ceil(r_max))
    canvas_size = 2 * max_radius_int + 1
    cart_img = np.zeros((canvas_size, canvas_size), dtype=polar_img.dtype)
    
    # Shift center to the local canvas coordinates
    local_cx = max_radius_int
    local_cy = max_radius_int
    
    for y in range(canvas_size):
        for x in range(canvas_size):
            dx = x - local_cx
            dy = y - local_cy
            r = np.sqrt(dx**2 + dy**2)
            
            # Check if point falls within the r_min and r_max bounds
            if r_min <= r <= r_max:
                theta = np.arctan2(dy, dx)
                if theta < 0:
                    theta += 2 * np.pi
                
                # Map continuous r and theta to discrete polar image indices
                r_idx = int((r - r_min) / (r_max - r_min) * (num_radii - 1))
                theta_idx = int(theta / (2 * np.pi) * (num_thetas - 1))
                
                r_idx = np.clip(r_idx, 0, num_radii - 1)
                theta_idx = np.clip(theta_idx, 0, num_thetas - 1)
                
                cart_img[y, x] = polar_img[r_idx, theta_idx]
                
    return cart_img
