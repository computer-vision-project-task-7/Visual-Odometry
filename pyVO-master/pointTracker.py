import numpy as np
import cv2
from pyflann import *
from typing import Tuple, List
from math import sin, cos, pi, sqrt
from collections import defaultdict
import time

flann = FLANN()

def is_invertible(a):
	return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def get_warped_patch(img: np.ndarray, patch_size: int,
					 x_translation: float, y_translation: float, theta) -> np.ndarray:

	"""
	Returns a warped image patch.     W(x;p) function

	:param img:             Original image.
	:param patch_size:      The size of the patch. Should be a odd number.
	:param x_translation:   The x position of the patch center.
	:param y_translation:   The y position of the patch center.
	:param theta:           The rotation of the patch in radians.

	:return:                The warped image patch.
	"""
	patch_half_size = patch_size//2
	c = cos(-theta)
	s = sin(-theta)
	t = np.array([[c, s, float((-c - s) * patch_half_size + x_translation)],
				  [-s, c, float((s - c) * patch_half_size + y_translation)]])
	
	return cv2.warpAffine(img, t, (patch_size, patch_size), flags=cv2.WARP_INVERSE_MAP)


class KLTTracker:

	def __init__(self, initial_position: np.ndarray, origin_image: np.ndarray, patch_size, tracker_id):
		assert patch_size >= 3 and patch_size % 2 == 1, f'patch_size must be 3 or greater and be a odd number, is {patch_size}'
		# koordinatene til features vi tracker
		self.initialPosition = initial_position

		self.translationX = 0.0
		self.translationY = 0.0
		self.theta = 0.0

		self.positionHistory = [(self.pos_x, self.pos_y, self.theta)]
		self.trackerID = tracker_id
		self.patchSize = patch_size
		self.patchHalfSizeFloored = patch_size // 2

		pos_x, pos_y = initial_position
		image_height, image_width = origin_image.shape
		assert self.patchHalfSizeFloored <= pos_x < image_width - self.patchHalfSizeFloored \
			   and self.patchHalfSizeFloored <= pos_y < image_height - self.patchHalfSizeFloored, \
			f'Point is to close to the image border for the current patch size, point is {initial_position} and patch_size is {patch_size}'
		self.trackingPatch = origin_image[pos_y - self.patchHalfSizeFloored:pos_y + self.patchHalfSizeFloored + 1,
										  pos_x - self.patchHalfSizeFloored:pos_x + self.patchHalfSizeFloored + 1]
		self.visualizeColor = np.random.randint(0, 256, 3, dtype=int)
		self.patchBorder = sqrt(2*patch_size**2) + 1

	@property
	def pos_x(self):
		return float(self.initialPosition[0] + self.translationX)

	@property
	def pos_y(self):
		return float(self.initialPosition[1] + self.translationY)

	def track_new_image(self, img: np.ndarray, img_grad: np.ndarray, max_iterations: int,
						min_delta_length=2.5e-2, max_error=0.0350) -> int: #max_error=0.035
		"""
		Tracks the KLT tracker on a new grayscale image. You will need the get_warped_patch function here.

		**Objective:**
		Tracking KLT tracker, whom which calculates how the new image (frame n) must
		be transformed to have same values as last frames (frame n-1) image.

		:param img:              The image.
		:param img_grad:         The image gradient.
		:param max_iterations:   The maximum number of iterations to run.
		:param min_delta_length: The minimum length of the delta vector.


		If the length is shorter than this, then the optimization should stop.

		:param max_error:    The maximum error allowed for a valid track.

		:return:         0 when track is successful,
						 1 any point of the tracking patch is outside the image
						 2 if a singular hessian is encountered
						 3 if the final error is larger than max_error.
		"""
	
		T = self.trackingPatch
		cv2.imshow('Template', cv2.resize(T*255,(270,270)).astype(np.uint8))
		
		for iteration in range(max_iterations):	
			#t = time.time()
			p = np.array([self.pos_x, self.pos_y, self.theta])	
			jac = np.zeros((self.patchSize, self.patchSize, 2, 3)) #bruk mgrid
			jac[:,:,0,0] = 1
			jac[:,:,0,1] = 0
			jac[:,:,1,0] = 0
			jac[:,:,1,1] = 1
			grid_x, grid_y = np.mgrid[-self.patchHalfSizeFloored:self.patchHalfSizeFloored+1,
			 -self.patchHalfSizeFloored:self.patchHalfSizeFloored+1]
			jac[:,:,0,2] = -grid_x * sin(self.theta) + grid_y * cos(self.theta)
			jac[:,:,1,2] = grid_x * cos(self.theta) - grid_y * sin(self.theta)
			# -----finne delta_p--------------
			grad = get_warped_patch(img_grad, self.patchSize, p[0], p[1], p[2]).reshape(27,27,2,1).transpose(0,1,3,2)
			
			I_jac = grad @ jac
			H = I_jac.transpose(0,1,3,2) @ I_jac
			H = np.sum(H, axis=(0,1))
			
			if is_invertible(H) == False:	# check if Hessian is singular (if not invertible => singular)
				return 2
			# hessian invertert
			H_inv = np.linalg.inv(H)
			
			# I(W(x;p))
			I_W = get_warped_patch(img, self.patchSize, p[0], p[1], p[2])
			
			cv2.imshow('I_W', cv2.resize(I_W*255,(270,270)).astype(np.uint8))
			# sum( T-I(w(x;p)))
			T_IW = (T-I_W).reshape(27, 27, 1, 1)
			cv2.imshow('T_IW', cv2.resize(255*np.abs(T_IW).reshape(27,27),(270,270)).astype(uint8))
			cv2.waitKey(5)
			delta_p = H_inv @ np.sum((I_jac * T_IW), axis=(0,1)).T

			# update trans_x, trans_y, theta
			self.translationX += delta_p[0]
			self.translationY += delta_p[1]
			self.theta        += 0.1 * delta_p[2]
			#print(time.time() - t)
			#check if points on the patch are outside the image
			if (self.pos_x-self.patchHalfSizeFloored <= 0 and self.pos_x+self.patchHalfSizeFloored >= img.shape[1]):
				if (self.pos_y-self.patchHalfSizeFloored <=0 and self.pos_y+self.patchHalfSizeFloored >= img.shape[0]):
					return 1
			# if length og delta_p is less than min_delta_length, stop optimazation
			if(np.linalg.norm(delta_p) < min_delta_length):
				break

		# Add new point to positionHistory to visualize tracking
		self.positionHistory.append((self.pos_x, self.pos_y, float(self.theta)))
		#print(self.pos_x, self.pos_y, float(self.theta))
		print('max error', np.sum(np.abs(T_IW)))
		if np.sum(np.abs(T_IW)) < max_error:
			# return 0 if error = ok, length(delta_p) = ok in max_iterations
			return 0
		else:
			return 3

class PointTracker:

	def __init__(self, max_points=5, tracking_patch_size=27):
		self.maxPoints = max_points
		self.trackingPatchSize = tracking_patch_size
		self.currentTrackers = []
		self.nextTrackerId = 0


	def visualize(self, img: np.ndarray, draw_id=False):
		img_vis = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
		for klt in self.currentTrackers:
			x_pos = int(round(klt.pos_x))
			y_pos = int(round(klt.pos_y))
			length = 20
			x2_pos = int(round(x_pos + length * cos(-klt.theta + pi / 2)))
			y2_pos = int(round(y_pos - length * sin(-klt.theta + pi / 2)))
			cv2.circle(img_vis, (x_pos, y_pos), 3, [int(c) for c in klt.visualizeColor], -1)
			cv2.line(img_vis, (x_pos, y_pos), (x2_pos, y2_pos), 0, thickness=1, lineType=cv2.LINE_AA)
			if draw_id:
				cv2.putText(img_vis, f'{klt.trackerID}', (x_pos+5, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 100, 0))

			if len(klt.positionHistory) >= 2:
				for i in range(len(klt.positionHistory)-1):
					x_from, y_from, _ = klt.positionHistory[i]
					x_to, y_to, _ = klt.positionHistory[i+1]
					cv2.line(img_vis, (int(round(x_from)), int(round(y_from))), (int(round(x_to)), int(round(y_to))), 0, thickness=1, lineType=cv2.LINE_AA)

		cv2.imshow("KLT Trackers", img_vis)



	def add_new_corners(self, origin_image: np.ndarray, points_and_response_list:  List[Tuple[float, np.ndarray]],
																						min_distance=13.0) -> None:

		assert len(points_and_response_list) > 0, 'points_list is empty'

		for i in range(len(points_and_response_list) - 1):
		 # Check that points_list is sorted from largest to smallest response value
			assert points_and_response_list[i][0] >= points_and_response_list[i + 1][0], 'points_list is not sorted'

		if len(self.currentTrackers) >= self.maxPoints:
		 # Dont do anything if we already have the maximum number of points
			return

		filtered_points = []

		image_height, image_width = origin_image.shape
		patch_border = sqrt(2 * self.trackingPatchSize ** 2) + 1
		for _, point in points_and_response_list:
		 # Filter out points to close to the image border
			pos_x, pos_y = point
			if patch_border <= pos_x < image_width - patch_border \
					and patch_border <= pos_y < image_height - patch_border:
				filtered_points.append(point)

		points = filtered_points
		filtered_points = []
		if len(self.currentTrackers) > 0:
		 # Filter out points to close to existing points
			current_points = [np.array([klt.pos_x, klt.pos_y]) for klt in self.currentTrackers]
			_, dists = flann.nn(np.array(current_points, dtype=np.int32), np.array(points, dtype=np.int32), 1)
			dists = np.sqrt(dists)
			filter_indices = np.arange(0, len(points))[dists >= min_distance]

			for i in filter_indices:
				filtered_points.append(points[i])
			points = filtered_points

		# Add at most enough points to bring us up to the max number of points
		number_of_points_to_add = min(len(points), self.maxPoints - len(self.currentTrackers))
		points = points[:number_of_points_to_add]

		for point in points:
			# origin_image = Template T(x)
			self.currentTrackers.append(KLTTracker(point, origin_image, self.trackingPatchSize, self.nextTrackerId))
			self.nextTrackerId += 1

	def track_on_image(self, img: np.ndarray, max_iterations=25) -> None:

		img_dx = cv2.Scharr(img, cv2.CV_64FC1, 1, 0)    # scharr likt sobel ( finner gradient )
		img_dy = cv2.Scharr(img, cv2.CV_64FC1, 0, 1)
		img_grad = np.stack((img_dx, img_dy), axis=-1)
		cv2.waitKey(1)
		lost_track = []
		tracker_return_values = defaultdict(int)
		for klt in self.currentTrackers:
			tracker_condition = klt.track_new_image(img, img_grad, max_iterations)
			#print(tracker_condition)
			tracker_return_values[tracker_condition] += 1
			if tracker_condition != 0:
				lost_track.append(klt)

		print(f"Tracked frame - remained: {tracker_return_values[0]}, hit border: {tracker_return_values[1]}, "
			  f"singular_hessian: {tracker_return_values[2]}, large error: {tracker_return_values[3]}")

		for klt in lost_track:
			self.currentTrackers.remove(klt)

	def get_position_with_id(self) -> Tuple[np.ndarray, np.ndarray]:
		n_points = len(self.currentTrackers)
		ids = np.empty(n_points, dtype=np.int32)
		positions = np.empty((2, n_points), dtype=np.float64)

		for i, klt in enumerate(self.currentTrackers):
			ids[i] = klt.trackerID
			positions[:, i] = (klt.pos_x, klt.pos_y)

		return ids, positions
