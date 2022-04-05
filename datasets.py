import os
import os.path
import sys

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import pickle
import random
import numpy as np
import pandas as pd

from copy import deepcopy
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from matplotlib import pyplot as plt

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def show(img):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tp = img.detach().cpu().numpy()
    tp = np.transpose(tp, [1, 2, 0])
    ax.imshow(tp)
    plt.show()

class FlowerTextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', transform=None, fixed_sentence_id=None):
        super(FlowerTextDataset, self).__init__()
        self.transform = transform
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.fixed_sentence_id = fixed_sentence_id

        self.data = []
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)

        self.test_class_id  = self.load_class_id(os.path.join(data_dir, 'test'))
        self.train_class_id = self.load_class_id(os.path.join(data_dir, 'train'))

        if split == 'train':
            self.class_id = self.train_class_id
        else:
            self.class_id = self.test_class_id

        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

    @staticmethod
    def __getImages(img_path, mask_path, transform=None, normalize=None):
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        # fimg full image
        fimg = deepcopy(img)  # FIXME why use additional np.array
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)

        cimg = deepcopy(img)
        if transform is not None:
            cimg = transform(cimg)

        cimgs = normalize(transforms.Resize((128, 128))(cimg))
        cimgsx64 = normalize(transforms.Resize((64, 64))(cimg))
        # show(cimgs)

        # We use full image to get background patches

        # We resize the full image to be 126 X 126 (instead of 128 X 128)  for the full coverage of the input (full) image by
        # the receptive fields of the final convolution layer of background discriminator

        crop_width = 128
        fimg = transforms.Resize((int(crop_width * 76 / 64), int(crop_width * 76 / 64)))(fimg)
        mask = transforms.Resize((int(crop_width * 76 / 64), int(crop_width * 76 / 64)))(mask)
        fw, fh = fimg.size

        # random cropping
        crop_start_x = np.random.randint(fw - crop_width)
        crop_start_y = np.random.randint(fh - crop_width)

        fimg = fimg.crop([crop_start_x, crop_start_y, crop_start_x + crop_width, crop_start_y + crop_width])
        mask = mask.crop([crop_start_x, crop_start_y, crop_start_x + crop_width, crop_start_y + crop_width])

        # random flipping
        random_flag = np.random.randint(2)
        if random_flag == 0:
            fimg = fimg.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        fimg = normalize(fimg)
        mask = transforms.ToTensor()(mask)

        return fimg, cimgs, cimgsx64, mask

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames, class_id):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text_c10/class_%05d/%s.txt' % (data_dir, class_id[i], filenames[i][filenames[i].index('/')+1:])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names, self.train_class_id)
            test_captions = self.load_captions(data_dir, test_names, self.test_class_id)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir):
        with open(data_dir + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding='bytes')
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((18, 1), dtype='int64')
        x_len = num_words
        if num_words <= 18:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:18]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = 18
        return x, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        key = key[:key.index('/')] + str(self.class_id[index]) + key[key.index('/'):]
        img_name  = '%s/jpg/%s.jpg' % (self.data_dir, key[key.index('/')+1:])
        mask_path = '%s/mask/%s.png' % (self.data_dir, key[key.index('/')+1:].replace('image', 'segmim'))
        fimg, cimgs, cimgsx64, object_mask = FlowerTextDataset.__getImages(img_name, mask_path, self.transform, normalize=self.norm)
        # random select a sentence

        sentence_id = self.fixed_sentence_id if self.fixed_sentence_id is not None\
            else random.randint(0, 9)

        text_index = index * 10 + sentence_id
        text, text_len = self.get_caption(text_index)

        return fimg, cimgs, text, text_len, cimgsx64, object_mask, key, sentence_id

    def __len__(self):
        return len(self.filenames)

class BirdTextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', fixed_sentence_id = None, transform=None):
        super(BirdTextDataset, self).__init__()
        self.transform = transform
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.fixed_sentence_id = fixed_sentence_id
        self.data = []
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.bbox = self.load_bbox()
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        self.class_id = self.load_class_id(self.split_dir, len(self.filenames))

        self.cls_set = {}
        cnt = 0
        for id in self.class_id:
            if not id in self.cls_set:
                self.cls_set[id] = cnt
                cnt += 1

    @staticmethod
    def __getImages(img_path, bbox=None, transform=None, normalize=None):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        extend_len = 16
        r = int(np.maximum(bbox[2], bbox[3]) * 0.5)
        #r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        rand_x = np.random.randint(0, extend_len + 1)
        rand_y = np.random.randint(0, extend_len + 1)
        y1 = np.maximum(0, center_y - r - rand_y)
        y2 = np.minimum(h, center_y + r + (extend_len - rand_y))
        x1 = np.maximum(0, center_x - r - rand_x)
        x2 = np.minimum(w, center_x + r + (extend_len - rand_x))
        #y1 = np.maximum(0, center_y - r)
        #y2 = np.minimum(h, center_y + r)
        #x1 = np.maximum(0, center_x - r)
        #x2 = np.minimum(w, center_x + r)

        # fimg full image
        fimg = deepcopy(img)  # FIXME why use additional np.array
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)
        # cimg cropped img
        cimg = img.crop([x1, y1, x2, y2])


        if transform is not None:
            cimg = transform(cimg)

        cimgs = normalize(transforms.Resize((128, 128))(cimg))
        cimgsx64 = normalize(transforms.Resize((64, 64))(cimg))
        # show(cimgs)

        # We use full image to get background patches

        # We resize the full image to be 126 X 126 (instead of 128 X 128)  for the full coverage of the input (full) image by
        # the receptive fields of the final convolution layer of background discriminator

        crop_width = 128
        fimg = transforms.Resize((int(crop_width * 76 / 64), int(crop_width * 76 / 64)))(fimg)
        fw, fh = fimg.size

        # random cropping
        crop_start_x = np.random.randint(fw - crop_width)
        crop_start_y = np.random.randint(fh - crop_width)

        fimg = fimg.crop([crop_start_x, crop_start_y, crop_start_x + crop_width, crop_start_y + crop_width])

        # bbox warp
        warped_x1 = bbox[0] * fw / w
        warped_y1 = bbox[1] * fh / h
        warped_x2 = warped_x1 + (bbox[2] * fw / w)
        warped_y2 = warped_y1 + (bbox[3] * fh / h)

        warped_x1 = min(max(0, warped_x1 - crop_start_x), crop_width)
        warped_y1 = min(max(0, warped_y1 - crop_start_y), crop_width)
        warped_x2 = max(min(crop_width, warped_x2 - crop_start_x), 0)
        warped_y2 = max(min(crop_width, warped_y2 - crop_start_y), 0)

        # random flipping
        random_flag = np.random.randint(2)
        if random_flag == 0:
            fimg = fimg.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_x1 = crop_width - warped_x2
            flipped_x2 = crop_width - warped_x1
            warped_x1 = flipped_x1
            warped_x2 = flipped_x2

        fimg = normalize(fimg)
        mask = torch.zeros_like(fimg)
        mask[:, int(warped_y1):int(warped_y2), int(warped_x1):int(warped_x2)] = 1.0

        #warped_bbox = np.array([warped_y1, warped_x1, warped_y2, warped_x2], dtype=np.float)
        return fimg, cimgs, cimgsx64, mask

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='bytes')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((25, 1), dtype='int64')
        x_len = num_words
        if num_words <= 25:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:25]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = 25
        return x, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        bbox = self.bbox[key]
        data_dir = '%s/CUB_200_2011' % self.data_dir
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        fimg, cimgs, cimgsx64, object_mask = BirdTextDataset.__getImages(img_name, bbox, self.transform, normalize=self.norm)
        
        sentence_id = self.fixed_sentence_id if self.fixed_sentence_id is not None\
            else random.randint(0, 9)

        text_index = index * 10 + sentence_id
        text, text_len = self.get_caption(text_index)
        return fimg, cimgs, text, text_len, cimgsx64, object_mask, key, sentence_id, cls_id

    def __len__(self):
        return len(self.filenames)