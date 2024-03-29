import time
import json
import os
import json
from io import BytesIO

import base64
import boto3
import math
import io
import cv2
import numpy as np

# Grab environment variables
TEST_ID = os.environ['TEST_ID']
DETECTION_ENDPOINT_NAME = os.environ['DETECTION_ENDPOINT_NAME']
CLASSIFICATION_ENDPOINT_NAME = os.environ['CLASSIFICATION_ENDPOINT_NAME']
UPLOAD_BUCKET = os.environ['UPLOAD_BUCKET']
DOWNLOAD_BUCKET = os.environ['DOWNLOAD_BUCKET']
READ_IMAGE_BUCKET = os.environ['READ_IMAGE_BUCKET']
KIT_DATA_JSON_PATH = os.environ['KIT_DATA_JSON_PATH']

MAX_HEIGHT = int(os.environ['MAX_HEIGHT'])
MEMBRANE_BOX_THRESHOLD = float(os.environ['MEMBRANE_BOX_THRESHOLD'])
KIT_BOX_THRESHOLD = float(os.environ['KIT_BOX_THRESHOLD'])
MEMBRANE_MASK_THRESHOLD = float(os.environ['MEMBRANE_MASK_THRESHOLD'])
KIT_MASK_THRESHOLD = float(os.environ['KIT_MASK_THRESHOLD'])
MEMBRANE_LOCALIZATION_THRESHOLD = float(os.environ['MEMBRANE_LOCALIZATION_THRESHOLD'])
ANGLE_CALCULATION_METHOD = str(os.environ['ANGLE_CALCULATION_METHOD'])
ANGLE_THRESHOLD = int(os.environ['ANGLE_THRESHOLD'])
CLASSIFICATION_THRESHOLDS = eval(str(os.environ['CLASSIFICATION_THRESHOLDS']))

ACCESS_KEY_ID = os.environ['ACCESS_KEY_ID']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']

# Grab client for running model inference
runtime = boto3.client('runtime.sagemaker')
# Grab client for uploading files to S3 bucket
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
# Get kit information
KIT_DATA_FILE = s3.get_object(Bucket=DOWNLOAD_BUCKET, Key=KIT_DATA_JSON_PATH)['Body']
KIT_DATA = json.load(KIT_DATA_FILE)


def lambda_handler(event, context):
	# Get current time
	start_time = time.strftime('%A %B, %d %Y %H:%M:%S')
	print('Received event: ' + json.dumps(event, indent=2))
	
	# Read in data
	data = json.loads(json.dumps(event))
	
	if 'input_type' in data:
		input_type = data['input_type']
	else:
		input_type = 'url'
	
	# As opposed to reading a b64 string (as shown above), now we read an S3 URL where the image is
	if input_type == 'url':
		image_url = data['image']
		try:
			initial_image_file = s3.get_object(Bucket=READ_IMAGE_BUCKET, Key=image_url)['Body']
			initial_image = file_to_array(image_file=initial_image_file)
		except:
			return {'statusCode': 404, 'body': 'The specified image URL could not be found!'}
		
		filename = os.path.splitext(os.path.split(image_url)[-1])[0]
		# Set the global ID for the S3 folder where the images will be saved
		splitter = ' | '
		global_id = '%s%s%s' % (str(start_time), splitter, str(filename))
	elif input_type == 'base64':
		base64string = data['image']
		# Convert base64string to NumPy image array and get image dimensions
		initial_image = string_to_array(image_string=base64string)
		global_id = '%s' % str(start_time)
	
	# Save original resolution for the classification step -- to use as much pixel information as we can 
	initial_image_original_resolution = initial_image.copy()
	
	# Resize the initial image for further operations regarding detection -- if height > MAX_HEIGHT
	if initial_image.shape[0] > MAX_HEIGHT:
		new_width = int((MAX_HEIGHT / initial_image.shape[0]) * initial_image.shape[1])
		initial_image = cv2.resize(initial_image, (new_width, MAX_HEIGHT))
	
	# Record the pre-processed height, width, and num. channels of the acquired image
	H, W, C = initial_image.shape 
	
	print(H, W, C)
	
	if C == 4:
		initial_image = initial_image[:, :, :-1]
	
	# Upload initial data
	upload_file(global_id=global_id, filename='initial_image.jpg', data=array_to_bytes(image_array=initial_image))
	
	# Run detection (NOTE: Pass image bytes)
	try:
		detection_response = runtime.invoke_endpoint(EndpointName=DETECTION_ENDPOINT_NAME,
													 ContentType='application/x-image',
													 Body=array_to_bytes(image_array=initial_image))
	except:
		return {'statusCode': 100, 'body': 'Object detection failed!'} 
	detection_response = json.loads(detection_response['Body'].read())
	
	labels = [prediction['label'] for prediction in detection_response]
	scores = [eval(prediction['score']) for prediction in detection_response]
	boxes = np.array([eval(prediction['box']) for prediction in detection_response])
	masks = np.array([eval(prediction['mask']) for prediction in detection_response])
	print('Labels: %s \nScores: %s' % (str(labels), str(scores)))
	
	try:
		kit_location, membrane_location = labels.index('kit'), labels.index('membrane')
		kit_score, membrane_score = scores[kit_location], scores[membrane_location]
	except:
		# Either kit or membrane is missing from the prediction or the image
		return {'statusCode': 101, 'body': 'Either kit or membrane is missing from the prediction or the image!'}
		
	if kit_score < KIT_BOX_THRESHOLD or membrane_score < MEMBRANE_BOX_THRESHOLD:
		# Either kit or membrane has low confidence!
		return {'statusCode': 102, 'body': 'Either kit or membrane has low confidence!'}
	
	# Get and compute masks
	kit_box, membrane_box = boxes[kit_location, :], boxes[membrane_location, :]
	kit_mask, membrane_mask = masks[kit_location, :, :, :], masks[membrane_location, :, :, :]
	
	# Segment binary masks for membrane and kit via the set threshold and convert to int dtype
	kit_mask[kit_mask >= KIT_MASK_THRESHOLD] = 1.
	kit_mask[kit_mask < KIT_MASK_THRESHOLD] = 0.
	kit_mask = np.array(np.concatenate([kit_mask.reshape((H, W, 1)) * 255] * 3, axis=-1), dtype=np.uint8)
	membrane_mask[membrane_mask >= MEMBRANE_MASK_THRESHOLD] = 1.
	membrane_mask[membrane_mask < MEMBRANE_MASK_THRESHOLD] = 0.
	membrane_mask = np.array(np.concatenate([membrane_mask.reshape((H, W, 1)) * 255] * 3, axis=-1), dtype=np.uint8)
	
	# Calculate rotation angle based on kit or membrane coordinates
	if ANGLE_CALCULATION_METHOD == 'kit_mask':
		lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord = compute_bquad(kit_mask)
	elif ANGLE_CALCULATION_METHOD == 'membrane_mask':
		lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord = compute_bquad(membrane_mask)
	left_angle = math.atan((leftbottom_coord[1] - lefttop_coord[1]) / (leftbottom_coord[0] - lefttop_coord[0])) * (180 / math.pi)
	right_angle = math.atan((rightbottom_coord[1] - righttop_coord[1]) / (rightbottom_coord[0] - righttop_coord[0])) * (180 / math.pi)
	angle = int(round((left_angle + right_angle) / 2))
	
	if abs(angle) >= ANGLE_THRESHOLD:
		# Image is heavily rotated!
		return {'statusCode': 103, 'body': 'Test kit is rotated!'}
		
	# Compute bounding box and bounding quadrilateral coordinates for kit and membrane masks
	kit_xmin, kit_xmax, kit_ymin, kit_ymax = compute_bbox(kit_mask)
	if (kit_xmax - kit_xmin) > (kit_ymax - kit_ymin):
		# Image is >= 45 degrees (e.g. 90) rotated
		return {'statusCode': 103, 'body': 'Test kit is >= 45 degrees rotated!'}
	  
	kit_lefttop_coord, kit_righttop_coord, kit_rightbottom_coord, kit_leftbottom_coord = compute_bquad(kit_mask)
	
	membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = compute_bbox(membrane_mask)
	membrane_lefttop_coord, membrane_righttop_coord, membrane_rightbottom_coord, membrane_leftbottom_coord = compute_bquad(membrane_mask)
	
	# Crop the image s.t. that it only contains the kit (and hence the membrane)
	kit_cropped_image = initial_image[kit_ymin: kit_ymax, kit_xmin: kit_xmax]
	
	# Update coordinates (i.e. subtract (kit_ymin, kit_xmin) from each)
	kit_lefttop_coord, membrane_lefttop_coord = kit_lefttop_coord - (kit_ymin, kit_xmin), membrane_lefttop_coord - (kit_ymin, kit_xmin)
	kit_righttop_coord, membrane_righttop_coord = kit_righttop_coord - (kit_ymin, kit_xmin), membrane_righttop_coord - (kit_ymin, kit_xmin)
	kit_rightbottom_coord, membrane_rightbottom_coord = kit_rightbottom_coord - (kit_ymin, kit_xmin), membrane_rightbottom_coord - (kit_ymin, kit_xmin)
	kit_leftbottom_coord, membrane_leftbottom_coord = kit_leftbottom_coord - (kit_ymin, kit_xmin), membrane_leftbottom_coord - (kit_ymin, kit_xmin)

	# Update xmin, xmax, ymin, and ymax for kit and membrane (i.e. subtract kit_ymin OR kit_xmin from each)
	kit_xmin, kit_xmax, kit_ymin, kit_ymax = 0, kit_cropped_image.shape[1] - 1, 0, kit_cropped_image.shape[0] - 1
	membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = membrane_xmin - kit_xmin, membrane_xmax - kit_xmin, membrane_ymin - kit_ymin, membrane_ymax - kit_ymin
	
	## Compute homography using kit coordinates ##
	# NOTE: Due to conventions of cv2's homography utilites, we will switch (y, x) around to be (x, y) in this section...
	# Take the source coordinates as the previously calculated ones
	kit_src_coords = np.array([kit_lefttop_coord[::-1], kit_righttop_coord[::-1], kit_rightbottom_coord[::-1], kit_leftbottom_coord[::-1]])
	
	# Get the ratio of the width to the height (i.e. aspect ratio)
	aspect_ratio = KIT_DATA['dimensions']['aspect_ratio']

	new_height, new_width = kit_cropped_image.shape[0], int(kit_cropped_image.shape[0] * aspect_ratio)
	# Take the destination coordinates as the far edges of the newly defined shape
	kit_dst_coords = np.array([[0, 0][::-1], [0, new_width-1][::-1], [new_height-1, new_width-1][::-1], [new_height-1, 0][::-1]])
	
	# Calculate homography
	homography_matrix, _ = cv2.findHomography(kit_src_coords, kit_dst_coords)
	
	# Warp source image to destination based on homography to get a correctly oriented image of the kit
	kit_warped_image = cv2.warpPerspective(kit_cropped_image, homography_matrix, (new_width, new_height))
	# Upload warped data
	upload_file(global_id=global_id, filename='kit_warped_image.jpg', data=array_to_bytes(image_array=kit_warped_image))
	
	# Apply the homography matrix to get the new membrane bquad coordinates
	membrane_lefttop_coord = apply_homography(src=membrane_lefttop_coord[::-1], homography_matrix=homography_matrix)[::-1]
	membrane_righttop_coord = apply_homography(src=membrane_righttop_coord[::-1], homography_matrix=homography_matrix)[::-1]
	membrane_rightbottom_coord = apply_homography(src=membrane_rightbottom_coord[::-1], homography_matrix=homography_matrix)[::-1]
	membrane_leftbottom_coord = apply_homography(src=membrane_leftbottom_coord[::-1], homography_matrix=homography_matrix)[::-1]

	# Get new kit and membrane bbox coordinates based on the warped test image and above bquad coordinates
	kit_xmin, kit_xmax, kit_ymin, kit_ymax = 0, kit_warped_image.shape[1] - 1, 0, kit_warped_image.shape[0] - 1
	membrane_xmin = min(membrane_lefttop_coord[1], membrane_leftbottom_coord[1])
	membrane_xmax = max(membrane_righttop_coord[1], membrane_rightbottom_coord[1])
	membrane_ymin = min(membrane_lefttop_coord[0], membrane_righttop_coord[0])
	membrane_ymax = max(membrane_leftbottom_coord[0], membrane_rightbottom_coord[0])
	
	# Make sure that all coordinates are >= 0
	membrane_xmin = max(0, membrane_xmin)
	membrane_xmax = max(0, membrane_xmax)
	membrane_ymin = max(0, membrane_ymin)
	membrane_ymax = max(0, membrane_ymax)
	
	# Calculated expected coordinates for membrane localization
	membrane_info = KIT_DATA['dimensions']['membrane']
	expected_membrane_xmin = int(kit_xmin + (kit_xmax * membrane_info['x']))
	expected_membrane_xmax = int(kit_xmin + (kit_xmax * (membrane_info['x'] + membrane_info['w'])))
	expected_membrane_ymin = int(kit_ymin + (kit_ymax * membrane_info['y']))
	expected_membrane_ymax = int(kit_ymin + (kit_ymax * (membrane_info['y'] + membrane_info['h'])))
	
	# Calculate the overlap based on the two sets of calculated coordinates for membranes (i.e. actual and expected)
	expected_membrane_zone = np.zeros(kit_warped_image.shape)[:, :, 0]
	expected_membrane_zone[expected_membrane_ymin: expected_membrane_ymax, expected_membrane_xmin: expected_membrane_xmax] = 1
	membrane_zone = np.zeros(kit_warped_image.shape)[:, :, 0]
	membrane_zone[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax] = 1
	overlap = expected_membrane_zone + membrane_zone
	overlap[np.where(overlap == 1)] = 0
	overlap_percentage = np.sum(overlap) / (np.sum(expected_membrane_zone) + np.sum(membrane_zone))

	if overlap_percentage < MEMBRANE_LOCALIZATION_THRESHOLD:
		# Membrane is not correctly localized, i.e. it is not approx. where we expect it to be!
		try:
			upload_file(global_id=global_id, filename='unlocalized_rotated_membrane.jpg', data=array_to_bytes(image_array=kit_warped_image[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax, :]))
		except:
			print('The calculated membrane bbox coordinates do not belong to the correct kit - most likely this means that there were 2 test kits in the image!')
		return {'statusCode': 104, 'body': 'Membrane was not correctly localized!'}
		
	# Keep homography calculated membrane just in case if |angle| >= ANGLE_THRESHOLD
	# homography_calculated_membrane = kit_warped_image[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax, :]
	## NOTE: We don't need the above line since we reject angles above the threshold in our cloud function!
	
	# NOTE: We don't have an sample inlet for the Quidel_Ag kit!
	
	## We have finished checking for localization and other constraints, so let's go back to the initial image and masks now
	# Get the rotated membrane using the angle and the membrane mask from before
	membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = compute_bbox(membrane_mask)
	
	## Upscale bbox coordinates to localize membrane in the original resolution image
	# Find scales for x-axis and y-axis
	y_scale = initial_image_original_resolution.shape[0] / membrane_mask.shape[0]
	x_scale = initial_image_original_resolution.shape[1] / membrane_mask.shape[1]
	# Apply scales to get bbox coordinates
	membrane_xmin_upscaled = int(membrane_xmin*x_scale) + 1
	membrane_xmax_upscaled = int(membrane_xmax*x_scale) - 1
	membrane_ymin_upscaled = int(membrane_ymin*y_scale) + 1
	membrane_ymax_upscaled = int(membrane_ymax*y_scale) - 1
	
	# Get rotated membrane
	# NOTE: We are going back to the original resolution membrane
	membrane = rotate_image(initial_image_original_resolution[membrane_ymin_upscaled: membrane_ymax_upscaled, membrane_xmin_upscaled: membrane_xmax_upscaled], -angle)
	# Upload rotated membrane
	upload_file(global_id=global_id, filename='rotated_membrane.jpg', data=array_to_bytes(image_array=membrane))
	
	# Compute the largest are bbox w/o any black zones that occurs due to angle
	membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = compute_bbox_for_rotated_rect_with_max_area(membrane.shape[0], membrane.shape[1], angle * (math.pi / 180))
	membrane = membrane[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax, :]
	
	# Apply white-balancing
	membrane = white_balance(membrane, KIT_DATA['ratio'])
	
	# Upload best rotated membrane
	upload_file(global_id=global_id, filename='best_rotated_membrane.jpg', data=array_to_bytes(image_array=membrane))
	
	# Run classification (NOTE: Pass image bytes)
	# NOTE: You don't have to rotate 90 degrees CCW as the model already takes care of this!
	try:
		classification_response = runtime.invoke_endpoint(EndpointName=CLASSIFICATION_ENDPOINT_NAME,
														  ContentType='application/x-image',
														  Body=array_to_bytes(image_array=membrane))
	except:
		return {'statusCode': 107, 'body': 'Image classification failed!'} 
	classification_response = json.loads(classification_response['Body'].read())
	print('Classification Response: ', classification_response)
													  
	zone_classification = eval(classification_response[0]['zone_classification'])
	confidence_scores = eval(classification_response[0]['confidence_scores'])
	diagnosis = eval(classification_response[0]['diagnosis'])
	
	positive_confidence_scores = [p[1] for p in confidence_scores]
	for k, P_pos in enumerate(positive_confidence_scores):
		if CLASSIFICATION_THRESHOLDS[0] <= P_pos <= CLASSIFICATION_THRESHOLDS[1]:
			return {'statusCode': 108, 'body': 'Image classification confidence is below the set threshold!'} 
	
	body = [zone_classification]
	if diagnosis == 0:
		body.append('NEGATIVE')
	elif diagnosis == 1:
		body.append('POSITIVE')
	else:
		body.append('INVALID')
												 
	return {'statusCode': 200, 'body': body}


def file_to_array(image_file):
	image_data = image_file.read()
	image_array =  np.frombuffer(image_data, dtype=np.uint8)
	image_array = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
	return image_array
	
	
def string_to_array(image_string):
	assert isinstance(image_string, str)
	image_data = base64.b64decode(image_string)
	image_array =  np.frombuffer(image_data, dtype=np.uint8)
	image_array = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
	return image_array
	

def array_to_bytes(image_array):
	assert isinstance(image_array, np.ndarray)
	_, image_bytes = cv2.imencode('.jpg', image_array)
	image_bytes = image_bytes.tobytes()
	
	return image_bytes


def upload_file(global_id, filename, data):
	with open('/tmp/%s' % filename, 'wb') as f:
		f.write(data)

	s3.upload_file('/tmp/%s' % filename, 
				   UPLOAD_BUCKET, 
				   'logs/%s/%s' % (global_id, filename))
				   

def rotate_image(image, angle):
	"""Function to rotate image in a specified (integer) angle"""
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


def compute_bbox_for_rotated_rect_with_max_area(h, w, angle):
	"""
	Given a rectangle of size wxh that has been rotated by 'angle' (in
	radians), computes the width and height of the largest possible axis-aligned rectangle 
	(maximal area) within the rotated rectangle and return bbox coordinates. Source:
	https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
	:param (int) h: height of the image
	:param (int) w: width of the image
	:param (float) angle: the angle of rotation in radians
	:return: (tuple) xmin, xmax, ymin, ymax -> bbox coords. of the rotated image
	"""
	if w <= 0 or h <= 0:
		return 0,0
	
	width_is_longer = w >= h
	side_long, side_short = (w,h) if width_is_longer else (h,w)
	
	# since the solutions for angle, -angle and 180-angle are all the same,
	# if suffices to look at the first quadrant and the absolute values of sin,cos:
	sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))

	## Commented this out because all cases I see are with 4 sides...
	# if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
	if False:
		# half constrained case: two crop corners touch the longer side,
		#   the other two corners are on the mid-line parallel to the longer line
		x = 0.5*side_short
		wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
	
	else:
		# fully constrained case: crop touches all 4 sides
		cos_2a = cos_a*cos_a - sin_a*sin_a
		wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
		
	w_bb = w*cos_a + h*sin_a
	h_bb = w*sin_a + h*cos_a
	inlet_horizontal = (w_bb - wr) / 2
	inlet_vertical = (h_bb - hr) / 2
	
	# Calculate coordinates and return them instead
	ymax = int(hr + inlet_vertical/2) - 1
	xmax = int(wr + inlet_horizontal/2) - 1
	ymin = round(round(inlet_vertical)/2) + 1
	xmin = round(round(inlet_horizontal)/2) + 1
	
	return xmin, xmax, ymin, ymax 


def apply_homography(src, homography_matrix):
	"""
	Function to apply homography H onto the source coordinate.
	:param (np.ndarray) src: source coordinate as (x, y)
	:param (np.ndarray) homography_matrix: denotes homography from (x, y) to (x', y') with shape (3, 3)
	:return: dst, destination coordinate as (x', y')
	"""
	x, y = src
	# Convert source coordinate space (i.e. cartesian) to homogenous coordinate space
	homogenous_src = np.asarray([[x], [y], [1]])
	# Compute destination coordinate in homogenous coordinate space
	homogenous_dst = np.dot(homography_matrix, homogenous_src)

	# We now have [[w*x'], [w*y'], [w]] as homogenous coordinate
	# We have to divide first two terms by w for cartesian coordinates of destination
	w = homogenous_dst[2][0]
	x_prime, y_prime = homogenous_dst[0][0] / w, homogenous_dst[1][0] / w

	# Convert float coordinates to integer
	dst = (int(round(x_prime)), int(round(y_prime)))
	return dst

def compute_bbox(mask):
	"""
	Function to compute the bounding box (i.e. rectangular) coordinates of a mask.
	:param (np.ndarray) mask: image mask with shape (H, W, C) and with values [0, 255]
	:return: bounding box (a.k.a bbox) coordinates -> (xmin, xmax, ymin, ymax)
	"""
	height, width, num_channels = mask.shape

	## (1) Get horizontal values ##
	left_edges = np.where(mask.any(axis=1), mask.argmax(axis=1), width + 1)
	# Flip horizontally to get right edges
	flip_lr = cv2.flip(mask, flipCode=1) 
	right_edges = width - np.where(flip_lr.any(axis=1), flip_lr.argmax(axis=1), width + 1)

	## (2) Get vertical values ##
	top_edges = np.where(mask.any(axis=0), mask.argmax(axis=0), height + 1)
	# Flip vertically to get bottom edges
	flip_ud = cv2.flip(mask, flipCode=0) 
	bottom_edges = height - np.where(flip_ud.any(axis=0), flip_ud.argmax(axis=0), height + 1)

	# Find the minimum and maximum values -> bbox coordinates
	xmin, xmax, ymin, ymax = left_edges.min(), right_edges.max(), top_edges.min(), bottom_edges.max()
	return xmin, xmax, ymin, ymax

def compute_bquad(mask):
	"""
	Function to compute the bounding quadrilateral (i.e. kite, parallelogram, trapezoid)
	coordinates of a mask
	:param (np.ndarray) mask: image mask with shape (H, W, C) and with values [0, 255]
	:return: bounding quadrilateral coordinates -> ((y1, x1), (y2, x2), (y3, x3), (y4, x4))
	"""
	def order_points(points):
		"""Function to return box-points in [lefttop, righttop, rightbottom, leftbottom] order"""
		# Convert from (x,y) representation to (y,x)
		points = np.array([point[::-1] for point in points])

		# Sort the points based on their y-coordinates
		points_ysorted = points[np.argsort(points[:, 0]), :]

		# Grab the bottommost and topmost points from the sorted y-coordinate points
		topmost, bottommost = points_ysorted[:2, :], points_ysorted[2:, :]

		# Sort the topmost coordinates according to their x-coordinates
		lefttop, righttop = topmost[np.argsort(topmost[:, 1]), :]

		# Apply the Pythagorean theorem
		distances = np.sum((righttop[np.newaxis] - bottommost)**2, axis=1)
		leftbottom, rightbottom = bottommost[np.argsort(distances)[::-1], :]

		return np.array([lefttop, righttop, rightbottom, leftbottom], dtype='int')

	# Valid mask coordinates are those that are not 0 and are 255
	mask_coords = np.argwhere(mask[:, :, 0] == 255)
	# Get the center (e.g (y/2, x/2)) of the mask
	center = np.mean(mask_coords, axis=0).astype('int')

	# Find contours and approximate a quadrilateral using the mask
	contours, _ = cv2.findContours(mask.copy()[:, :, 0].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea)
	rect = cv2.minAreaRect(contour)
	box = cv2.boxPoints(rect).astype('int')
	
	lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord = order_points(box)
	return lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord

def white_balance(images, ratio):
	"""Function for performing the white-balancing operation."""
	def rgb2ycbcr(im):
		xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
		ycbcr = im.dot(xform.T)
		ycbcr[:,:,[1,2]] += 128
		return ycbcr
		
	def ycbcr2rgb(im):
		xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
		rgb = im.astype(np.float)
		rgb[:,:,[1,2]] -= 128
		rgb = rgb.dot(xform.T)
		np.putmask(rgb, rgb > 255, 255)
		np.putmask(rgb, rgb < 0, 0)
		return np.uint8(rgb)
	
	data = np.asarray(images)
	refpos = (int(data.shape[0]*eval(ratio[0])[1]), int(data.shape[1]*eval(ratio[0])[0])) # top left corner
	refend = (int(data.shape[0]*eval(ratio[1])[1]), int(data.shape[1]*eval(ratio[1])[0]))
	sub = data[refpos[0]:refend[0],refpos[1]:refend[1]]
	
	ycbcr = rgb2ycbcr(data)
	ysub = rgb2ycbcr(sub)
	yc = list(np.mean(ysub[:,:,i]) for i in range(3))
	for i in range(1,3):
		ycbcr[:,:,i] = np.clip(ycbcr[:,:,i] + (128-yc[i]), 0, 255)
    
	rgb = ycbcr2rgb(ycbcr)
	# rgb = Image.fromarray(rgb)
    
	return rgb
