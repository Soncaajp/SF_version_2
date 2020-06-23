import os
import numpy as np
import mxnet as mx
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from utils.python_routines import read_affectnet_dict, add_rotations_affectnet, make_dir_dict, read_face_count
from utils.augmentation import get_augmentation_func
import math
from itertools import count

class AffectnetIter(mx.io.DataIter):
	def __init__(self, data_json_path, batch_size, train, img_size, detector,multiply_basic_ratio=1, provide_thresholds=False):
		super(AffectnetIter, self).__init__()
		files = os.listdir(data_json_path)
		if train == True:
			data = files[:int(len(files)*0.8)]
		else:
			data = files[int(len(files)*0.8):]
		self.label2img = data
		self.detector = detector
		# if not os.path.isdir(data_json_path):
		#     img2label, self.label2img = read_affectnet_dict(data_json_path)
		# else:
		#     img2label, self.label2img = make_dir_dict(data_json_path)





		# self.n_objects = data.shape[0]




		# faces_count = read_face_count()
		# if train:
		#     p_with_faces = set()
		#     for k in self.label2img.keys():
		#         # print(self.label2img[k])
		#         self.label2img[k] = [p for p in self.label2img[k] if (p[0] in faces_count) and (faces_count[p[0]] == 1)]
		#         for p in self.label2img[k]:
		#             p_with_faces.add(p[0])
		#     # print(type(p_with_faces), len(p_with_faces), type(p_with_faces[0]), p_with_faces[0])
		#     img2label = {k: v for k, v in img2label.items() if k in p_with_faces}
		#     img2label, self.label2img = add_rotations_affectnet(img2label, self.label2img, multiply_basic_ratio)

		self.provide_thresholds = provide_thresholds
		self.train = train
		self.batch_size = batch_size
		self.img_size = img_size
		# self.start = 0
		# self.end = batch_size -1
		self.position = 0
		# self.n_classes = len(self.label2img)
		# self.n_objects = int(self.n_classes * min(len(v) for k, v in self.label2img.items()) / multiply_basic_ratio)
		# print(self.n_objects, 'objects',
		#       self.n_objects // self.batch_size, 'batches per {} epoch'.format('train' if train else 'val'))
		# for k, v in self.label2img.items():
		#     print('{}: {}'.format(k, len(v)))
		# self.current_class = 0
		self.global_num_inst = 0
		self.used = []
		self.current = 0

		self.provide_data = [mx.io.DataDesc(
				name='data', shape=(batch_size, 3) + img_size)]
		if self.provide_thresholds:
			self.provide_label = [mx.io.DataDesc(name='softmax_label', shape=(batch_size,)),
								  mx.io.DataDesc(name='thresholds', shape=(batch_size,))]
		else:
			self.provide_label = [mx.io.DataDesc(name='softmax_label', shape=(batch_size,))]

		if self.train:
			self.aug = get_augmentation_func()

		# self.n_objects = int(self.n_objects / 10)
		# print('n_objects =', self.n_objects)
		# print('{} batches per epoch'.format(self.n_objects // self.batch_size))
		self.reset()


	def reset(self):
		self.objects_processed = 0
		if self.train:
			print('1')
			# for k in self.label2img.keys():
			#     np.random.shuffle(self.label2img[k])
			# self.label2img = self.label2img.sample(frac = 1)
		else:
			self.current_class = 0

	# def get_image(self, img_data):
	#     path = img_data[0]

	#     img = cv2.imread(path)
	#     if img is None:
	#         print('no img found:', img_data)
	#         return None
	#     img = img[:, :, ::-1]
	#     img = img[img_data[1]//2:(img_data[1]+img_data[3]+img.shape[0])//2,
	#               img_data[2]//2:(img_data[2]+img_data[4]+img.shape[1])//2]
	#     img = cv2.resize(img, self.img_size)
	#     if self.train:
	#         img = self.aug(img)
	#     img = np.moveaxis(img, -1, 0) / 255.
	#     # print(np.all(np.isfinite(img)))
	#     return img

	def preprocess_image(image):
		h, w = image.shape[:2]
		if w > h:
			diff_2 = (w - h) // 2
			image = cv2.copyMakeBorder(image, diff_2, diff_2, 0, 0, cv2.BORDER_CONSTANT, 0)
		elif h > w:
			diff_2 = (h - w) // 2
			image = cv2.copyMakeBorder(image, 0, 0, diff_2, diff_2, cv2.BORDER_CONSTANT, 0)

		image = cv2.resize(image, (224, 224))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image.transpose((2, 0, 1)) / 255

	def txt_parse(path, num):
		file = open(path, mode = 'r', encoding = 'utf-8-sig')
		lines = file.readlines()
		file.close()
		return (float(lines[num]))

	# def next(self):
	# 	if self.objects_processed + self.batch_size > self.n_objects:
	# 		raise StopIteration

	# 	xb = np.zeros(self.provide_data[0].shape)
	# 	yb = np.zeros(self.provide_label[0].shape)
	# 	# try parallel loop



	# 	starter_path = '/run/user/1000/gvfs/smb-share:server=as6210t-39a9,share=public/_WORK_FOLDER/EXTERNAL_DS/aff_wild/videos/train/'
	# 	for vid in self.label2img:
	# 		if self.current == 0:
	# 			self.current = vid
	# 		vidcap = cv2.VideoCapture(starter_path+vid)
	# 		for frame_id in count:
	# 			try:
	# 				if frame_id < self.position:
	# 					continue
	# 				success, picture = vidcap.read()
	# 				if not success:
	# 					break
	# 				self.detector.reset()
	# 				self.detector.store_data(picture, None)
	# 				res = self.detector.process_data()
	# 				x, y, w, h = [res[0][0]['key'] for key in 'xywh']
	# 				img = preprocess_image(picture[y:y+h, x:x+w])
	# 			except:
	# 				continue




	# 	for frame_id in count():
	# 	print(frame_id)
	# 	try:
	# 		if frame_id == 11001:
	# 			break
	# 		success, picture = vidcap.read()
	# 		if not success:
	# 			break
	# 		detector.reset()
	# 		detector.store_data(picture, None)
	# 		res = detector.process_data()
	# 		x, y, w, h = [res[0][0][key] for key in 'xywh']
	# 		img = preprocess_image(picture[y:y+h, x:x+w])
	# 		# cv2.imshow('pic', face)
	# 		# cv2.waitKey(0)
	# 		# print(int(res[0][0]['x'])+int(0.4*int(res[0][0]['w'])))
	# 		# print(int(res[0][0]['x']))
	# 		# img = get_image([picture, int(res[0][0]['y']), int(res[0][0]['x'])+int(2.5*int(res[0][0]['w'])), int(0.7*int(res[0][0]['h'])), int(res[0][0]['w'])])
	# 		# print(img)
	# 		# cv2.imshow('pic', img[0])
	# 		# cv2.waitKey(0)
	# 		# cv2.destroyAllWindows()
	# 		img = mx.nd.array(img[np.newaxis])
	# 		net.forward(mx.io.DataBatch(data=[img]))
	# 		out = net.get_outputs()[0][0].asnumpy()
	# 		temp.append(out[0])
	# 		if frame_id % 10 == 0:
	# 			pred.append(mean(temp))
	# 			temp = []
	# 			truth.append(txt_parse('/run/user/1000/gvfs/smb-share:server=as6210t-39a9,share=public/_WORK_FOLDER/EXTERNAL_DS/aff_wild/annotations/train/valence/172.txt', frame_id - 4))
	# 	except:
	# 		continue
	# 	vidcap.release()







	# 	if (self.position + self.batch_size > self.n_objects):
	# 		self.position = 0
	# 		print('New position =', self.position)


	# 	for i in range(self.batch_size):
	# 		row = self.label2img.iloc[self.position]
	# 		# print(row)
	# 		# print('I = ', i)
	# 		# print('Path = ', row['subDirectory_filePath'])
	# 		# print(math.isnan(row['face_x']))
	# 		while (math.isnan(row['face_x']) == True or math.isnan(row['face_y']) == True):
	# 			self.objects_processed += 1
	# 			self.position += 1
	# 			row = self.label2img.iloc[self.position]
	# 		img = self.get_image([starter_path + row['subDirectory_filePath'], int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])])
	# 		while row['valence'] == -2.0 or row['valence'] == 0. or img is None:
	# 			self.objects_processed += 1
	# 			self.position += 1
	# 			row = self.label2img.iloc[self.position]
	# 			while (math.isnan(row['face_x']) == True or math.isnan(row['face_y']) == True):
	# 				self.objects_processed += 1
	# 				self.position += 1
	# 				row = self.label2img.iloc[self.position]
	# 			# print('I = ', self.position)
	# 			# print('Path = ', row['subDirectory_filePath'])
	# 			img = self.get_image([starter_path + row['subDirectory_filePath'], int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])])
	# 		yb[i] = row['valence']
	# 		xb[i] = img
	# 		# print(img)
	# 		# print(row['valence'])
	# 		self.objects_processed += 1
	# 		self.global_num_inst += 1
	# 		self.position += 1



	# 	# # for i in range(self.batch_size):
	# 	# if self.end + self.batch_size > self.label2img.shape[0]:
	# 	#     self.end = self.label2img.shape[0]
	# 	# test = self.label2img.loc[self.start:self.end]
	# 	# count_bad = 0
	# 	# for index, row in test.iterrows():
	# 	#     if row['valence'] == -2.0:
	# 	#         count_bad += 1
	# 	# if self.end != self.label2img.shape[0]:
	# 	#     self.end += count_bad
	# 	# batch = self.label2img.loc[self.start:self.end]
	# 	# starter_path = '/home/nlab/Desktop/Affectnet/'
	# 	# print(batch.shape[0] - count_bad)
	# 	# i = 0
	# 	# for index, row in batch.iterrows():
	# 	#     if row['valence'] != -2.0:
	# 	#         img = self.get_image([starter_path + row['subDirectory_filePath'], int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])])
	# 	#         if img is not None:
	# 	#             # self.current_class = (self.current_class + 1) % self.n_classes
	# 	#             self.objects_processed += 1
	# 	#             img = self.get_image([starter_path + row['subDirectory_filePath'], int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])])
	# 	#             yb[i] = row['valence']
	# 	#             xb[i] = img
	# 	#             plt.imshow(img[0])
	# 	#             plt.show()
	# 	#             i +=1
	# 	#             self.objects_processed += 1
	# 	#             self.global_num_inst += 1

	# 	# self.start += (self.batch_size + count_bad)
	# 	# print('Self start:', self.start)
	# 	# self.end += self.batch_size
	# 	# print('Self end:', self.end)   
	# 	#     # img = self.get_image(self.label2img[self.current_class][int(self.objects_processed / self.n_classes)])
	# 	#     # while img is None:
	# 	#     #     self.current_class = (self.current_class + 1) % self.n_classes
	# 	#     #     self.objects_processed += 1
	# 	#     #     img = self.get_image(self.label2img[self.current_class][int(self.objects_processed / self.n_classes)])
	# 	#     # yb[i] = self.current_class
	# 	#     # xb[i] = img
	# 	#     # self.current_class = (self.current_class + 1) % self.n_classes
	# 	#     # self.objects_processed += 1
	# 	#     # self.global_num_inst += 1

	# 	labels = [mx.nd.array(yb)]
	# 	# if self.provide_thresholds:
	# 	#     labels.append(mx.nd.zeros_like(labels[0]))

	# 	# print('xb shape {}\t labels shape, {}'.format(xb.shape, labels.shape))

	# 	return mx.io.DataBatch([mx.nd.array(xb)], labels,
	# 					 provide_data=self.provide_data,
	# 					 provide_label=self.provide_label)



class FECIter(mx.io.DataIter):
	def __init__(self, img_path, data_csv_path, batch_size, train, img_size, provide_sm_labels=False):
		super(FECIter, self).__init__()
		self.provide_sm_labels = provide_sm_labels
		self.train = train
		self.batch_size = batch_size
		self.img_size = img_size
		self.img_path = img_path
		self.data = pd.read_csv(data_csv_path)
		self.n_triplets = int(len(self.data))
		# if self.train:
		#     self.n_triplets = int(len(self.data) / 30)
		# else:
		#     self.n_triplets = int(len(self.data) / 10)

		print(self.n_triplets, 'triplets',
			  self.n_triplets // self.batch_size, 'batches per {} epoch'.format('train' if train else 'val'))

		self.provide_data = [mx.io.DataDesc(name='data', shape=(3 * batch_size, 3) + img_size)]
		self.provide_label = ([mx.io.DataDesc(name='softmax_label', shape=(3 * batch_size,)),
							   mx.io.DataDesc(name='thresholds', shape=(3 * batch_size,))]
							  if self.provide_sm_labels else
							  [mx.io.DataDesc(name='thresholds', shape=(3 * batch_size,))])

		self.global_num_inst = 0

		if self.train:
			self.aug = get_augmentation_func()

		self.triplets_perm = np.random.permutation(self.n_triplets)
		self.reset()

	def reset(self):
		self.tripltes_processed = 0
		if self.train:
			self.triplets_perm = np.random.permutation(self.n_triplets)

	@staticmethod
	def to_square(img):
		h, w = img.shape[:2]
		a = min(w, h)
		return img[int((h-a) / 2):int((h-a) / 2) + a, int((w-a) / 2):int((w-a) / 2) + a]

	def get_image(self, img_data):
		path = os.path.join(self.img_path, 'train' if self.train else 'test', img_data)
		img = cv2.imread(path)
		if img is None:
			return None
		img = img[:, :, ::-1]
		img = cv2.resize(self.to_square(img), self.img_size)
		if self.train:
			img = self.aug(img)
		img = np.moveaxis(img, -1, 0) / 255.
		# if np.max(np.abs(img)) < 0.2:
		#     print(np.max(np.abs(img)), img_data)
		# print(img.min(), img.max())
		# plt.imshow(np.moveaxis(img, 0, -1))
		# plt.show()
		return img

	def next(self):
		if self.tripltes_processed + self.batch_size > self.n_triplets // 10:
			raise StopIteration
		xb = np.zeros(self.provide_data[0].shape)
		thresholds = np.zeros(self.batch_size)
		for i in range(self.batch_size):
			threshold = 0.1 if self.data.loc[self.triplets_perm[self.tripltes_processed], 'label'] == 'ONE_CLASS_TRIPLET' else 0.2
			for triplet_img_id in range(3):
				path = self.data.loc[self.triplets_perm[self.tripltes_processed], 'im{}'.format(triplet_img_id + 1)]
				img = self.get_image(path)
				if img is not None:
					xb[self.batch_size * triplet_img_id + i] = img
				else:
					threshold = -100
			thresholds[i] = threshold
			self.tripltes_processed += 1
			self.global_num_inst += 1
			# img = self.get_image(self.label2img[self.current_class][int(self.objects_processed / self.n_classes)])
			# while img is None:
			#     self.current_class = (self.current_class + 1) % self.n_classes
			#     self.objects_processed += 1
			#     img = self.get_image(self.label2img[self.current_class][int(self.objects_processed / self.n_classes)])
			# yb[i] = self.current_class
			# xb[i] = img
			# self.current_class = (self.current_class + 1) % self.n_classes
			# self.objects_processed += 1
			# self.global_num_inst += 1

		thresholds = mx.nd.array(np.tile(thresholds, 3))
		labels = [-mx.nd.ones_like(thresholds), thresholds] if self.provide_sm_labels else [thresholds]
		# print('xb shape {}\t thresholds shape, {}'.format(xb.shape, thresholds.shape))
		return mx.io.DataBatch([mx.nd.array(xb)], labels,
							   provide_data=self.provide_data,
							   provide_label=self.provide_label)
