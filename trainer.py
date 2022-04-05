from __future__ import print_function
from six.moves import range
import numpy as np
import os
import time
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from skimage.io import imsave

from miscc.config import cfg
from miscc.utils import mkdir_p
import pickle

from model import Generator, ConditionalImageDiscriminator, BackgroundDiscriminator
from loss import cosine_similarity, func_attention, sent_loss, words_loss, KL_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_network(gpus):
    count = 0

    netG = Generator(z_dim=256, t_dim=256, s_dim=256, noise_stage_num = 1)
    netG.apply(weights_init)

    netBGD = BackgroundDiscriminator()
    netBGD.apply(weights_init)

    netDs = [
        ConditionalImageDiscriminator(in_size = 64 * (2 ** i), text_dim = 256) \
            for i in range(2)
    ]
    for netD in netDs:
        netD.apply(weights_init)

    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.BGD != '':
        state_dict = torch.load(cfg.TRAIN.BGD)
        netBGD.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.BGD)

    if cfg.TRAIN.D != '':
        for i in range(len(netDs)):
            state_dict = torch.load(cfg.TRAIN.D + str(i) + '.pth')
            netDs[i].load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.D + str(i) + '.pth')

    if cfg.CUDA:
        netBGD.cuda()
        netG.cuda()
        for netD in netDs:
            netD.cuda()

    return netG, netBGD, netDs, count

def define_optimizers(netG, netBGD, netDs):
    optimizer_G = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_BGD = optim.Adam(netBGD.parameters(), lr=2e-4, betas=(0.5, 0.999))

    optimizer_Ds = []
    for netD in netDs:
        optimizer_Ds.append(
            optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        )

    return optimizer_G, optimizer_BGD, optimizer_Ds

def save_model(netG, netBGD, netDs, epoch, model_dir):
    for i in range(len(netDs)):
        torch.save(netDs[i].state_dict(), '%s/netD%s.pth' % (model_dir, str(i)))
    torch.save(netBGD.state_dict(), '%s/netBGD.pth' % (model_dir))
    torch.save(netG.state_dict(), '%s/netG_%d.pth' % (model_dir, epoch))
    print('Save G/Ds models.')

def save_img_results(fake_imgs, text, text_len, word_dict, count, image_dir):
    for i in range(len(fake_imgs)):
        fake_img = fake_imgs[i]
        vutils.save_image(fake_img.data, '%s/count_%09d_fake_samples%d.png' %(image_dir, count, i), normalize=True)

    text_log = ""
    ttext = text.detach().cpu().numpy()
    for i in range(text.shape[0]):
        for j in range(text_len[i]):
            text_log = text_log + word_dict[ttext[i][j]] + ' '
        text_log += '\n'

    with open('%s/count_%09d_text' %(image_dir, count), 'w') as f:
        f.write(text_log)

class FineGAN_trainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(0)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    @staticmethod
    def __prepare_data(data):
        full_image, imgs, text, text_len, imgsx64, obj_mask = data
        # sort data by the length in a decreasing order
        text_len, sorted_text_indice = torch.sort(text_len, 0, True)
        imgs = imgs[sorted_text_indice]
        full_image = full_image[sorted_text_indice]
        text = text[sorted_text_indice].squeeze()
        imgsx64 = imgsx64[sorted_text_indice]
        obj_mask = obj_mask[sorted_text_indice]

        text = text.cuda()
        imgs = imgs.cuda()
        text_len = text_len.cuda()
        full_image = full_image.cuda()
        imgsx64 = imgsx64.cuda()
        obj_mask = obj_mask.cuda()

        return full_image, imgs, text, text_len, imgsx64, obj_mask

    def optimize_BGD(self, D, opt, real, fake):
        D.zero_grad()
        batch_size = real.shape[0]

        real_bf_score, real_rf_score = D(real)
        fake_bf_score, fake_rf_score = D(fake.detach())
        real_labels = torch.ones_like(real_rf_score)
        fake_labels = torch.zeros_like(real_rf_score)
        #print(self.obj_mask.shape)
        patch_receptive_label = 1 - D.getPatchLevelLabel(self.obj_mask)

        loss_real_rf = nn.BCELoss(reduce=False)(real_rf_score, real_labels)  # Real/Fake loss for 'real background' (on patch level)
        loss_real_rf = torch.mul(loss_real_rf, patch_receptive_label)  # Masking output units which correspond to receptive fields which lie within the boundin box
        loss_real_rf = torch.sum(loss_real_rf) / torch.sum(patch_receptive_label)

        loss_real_bf = nn.BCELoss(reduce=False)(real_bf_score, patch_receptive_label)
        loss_real_bf = loss_real_bf.mean()

        #loss_fake_rf = (nn.Softplus()(fake_rf_score)).mean()
        loss_fake_rf = nn.BCELoss(reduce=False)(fake_rf_score, fake_labels)
        loss_fake_rf = loss_fake_rf.mean()
        
        loss = loss_real_bf + (loss_fake_rf + loss_real_rf) * cfg.TRAIN.BG_LOSS_WT

        loss.backward()
        opt.step()

        return loss

    def optimize_conditional_imgD(self, D, opt, real, fake, condition):
        opt.zero_grad()

        real_pred = D(real, condition)
        fake_pred = D(fake.detach(), condition)

        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        loss_real = nn.BCELoss()(real_pred, real_labels)
        loss_fake = nn.BCELoss()(fake_pred, fake_labels)

        loss = loss_real + loss_fake

        loss.backward()
        opt.step()

        return loss

    def train_Gnet(self, netG, netBGD, netDs, bg_img, fake_images, condition):
        netG.zero_grad()
        errG_total = 0

        # background
        bf_score, rf_score = netBGD(bg_img)
        real_labels = torch.ones_like(bf_score)
        errG_total += nn.BCELoss()(rf_score, real_labels) * cfg.TRAIN.BG_LOSS_WT + nn.BCELoss()(bf_score, real_labels)

        # object
        for i in range(len(netDs)):
            D_rf_score = netDs[i](fake_images[i], condition)
            real_labels = torch.ones_like(D_rf_score)
            errG_total += nn.BCELoss()(D_rf_score, real_labels)

        return errG_total

    def train(self, text_encoder, image_encoder, word_dict):
        lamb = 5.0 

        self.netG, self.netBGD, self.netDs, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizer_G, self.optimizer_BGD, self.optimizer_Ds = \
            define_optimizers(self.netG, self.netBGD, self.netDs)
        #with open('optimizers.pkl', 'rb') as f:
        #    self.optimizer_G, self.optimizer_BGD, self.optimizer_Ds = pickle.load(f)

        noise = Variable(torch.FloatTensor(self.batch_size, 256))
        self.match_labels = Variable(torch.LongTensor(range(self.batch_size)))

        if cfg.CUDA:
            noise = noise.cuda()
            image_encoder = image_encoder.cuda()
            text_encoder = text_encoder.cuda()
            self.match_labels = self.match_labels.cuda()

        print("Starting normal DGattGAN training..")
        count = start_count
        start_epoch = start_count // (self.num_batches)

        text_encoder.eval()
        image_encoder.eval()
        
        for epoch in range(600):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                self.fake_imgs = []
                self.real_imgs = []

                self.real_fimgs, self.real_cimgs, self.text, self.text_len, self.real_cimgsx64, self.obj_mask = FineGAN_trainer.__prepare_data(data)
                self.real_imgs.extend([self.real_cimgsx64, self.real_cimgs])

                hidden = text_encoder.init_hidden(self.batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(self.text, self.text_len, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (self.text == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                mask = mask.cuda()

                # Feedforward through Generator. Obtain stagewise fake images
                noise.data.normal_(0, 1)
                bg_image, fake_images, ob_masks, mu, log_var = self.netG(noise, sent_emb, words_embs, mask)
                
                self.fake_imgs.extend(fake_images)

                # Update Discriminator networks 
                errD_total = 0
                errD_total += self.optimize_BGD(self.netBGD, self.optimizer_BGD, self.real_fimgs, bg_image)
                for i in range(len(self.netDs)):
                    errD_total += self.optimize_conditional_imgD(self.netDs[i], self.optimizer_Ds[i], self.real_imgs[i], self.fake_imgs[i], sent_emb)

                # Update the Generator networks
                errG_total = self.train_Gnet(self.netG, self.netBGD, self.netDs, bg_image, fake_images, sent_emb)
                
                kl_loss = KL_loss(mu, log_var)
                errG_total += kl_loss
                
                region_features, cnn_code = image_encoder(fake_images[-1])
                w_loss0, w_loss1, _ = words_loss(region_features, words_embs, self.match_labels, self.text_len, None, self.batch_size)
                w_loss = (w_loss0 + w_loss1) * lamb

                s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, self.match_labels, None, self.batch_size)
                s_loss = (s_loss0 + s_loss1) * lamb
                errG_total += w_loss + s_loss

                errG_total.backward()
                self.optimizer_G.step()

                # ECA technique
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                count = count + 1

                if count % 5000== 0:
                    backup_para = copy_G_params(self.netG.cpu())
                    load_params(self.netG, avg_param_G)
                    save_model(self.netG, self.netBGD, self.netDs, count, self.model_dir)

                    self.fake_imgs = []

                    self.netG.cuda()
                    self.netG.eval()

                    with torch.set_grad_enabled(False):
                        bg_image, fake_images, ob_masks, mu, log_var = self.netG(noise, sent_emb, words_embs, mask)
                        self.fake_imgs.extend([bg_image])
                        self.fake_imgs.extend(fake_images)
                        self.fake_imgs.extend(ob_masks)
                        
                    self.netG.train()

                    save_img_results((self.fake_imgs + [self.real_cimgs]), self.text, self.text_len, word_dict, count, self.image_dir)
                    with open('optimizers.pkl', 'wb') as f:
                        pickle.dump((self.optimizer_G, self.optimizer_BGD, self.optimizer_Ds), f)
                    
                    self.netG = self.netG.cpu()
                    load_params(self.netG, backup_para)
                    self.netG.cuda()

                if step % 100 == 0:
                    print('%.5f %.5f' % (errD_total, errG_total))

            end_t = time.time()
            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Time: %.2fs
                      '''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data, errG_total.data,
                     end_t - start_t))

        save_model(self.netG, self.netBGD, self.netDs, count, self.model_dir)

class FineGAN_evaluator(object):
    
    @staticmethod
    def __prepare_data(data, gpu = 0):
        full_image, imgs, text, text_len, imgsx64, object_mask, key, sentence_id, cls_id = data
        # sort data by the length in a decreasing order
        text_len, sorted_text_indice = torch.sort(text_len, 0, True)
        
        imgs        = imgs[sorted_text_indice]
        imgsx64     = imgsx64[sorted_text_indice]
        full_image  = full_image[sorted_text_indice]
        object_mask = object_mask[sorted_text_indice]
        text        = text[sorted_text_indice].squeeze()
        sorted_key  = [key[i] for i in sorted_text_indice.numpy()]
        sentence_id = [sentence_id[i] for i in sorted_text_indice.numpy()]

        # send data to device
        text        = text.cuda()
        imgs        = imgs.cuda()
        imgsx64     = imgsx64.cuda()
        text_len    = text_len.cuda()
        full_image  = full_image.cuda()
        object_mask = object_mask.cuda()

        return full_image, imgs, text, text_len, imgsx64, object_mask, sorted_key, sentence_id, cls_id

    @staticmethod
    def __normalize(data):
        return (data - np.min(data))/(np.max(data) - np.min(data))

    @staticmethod
    def evaluate(text_encoder, data_loader, save_path, G_model_path, word_dict, batch_size, gpu = 0):
        torch.cuda.set_device(gpu)
        netG = Generator(z_dim=128, t_dim=256, s_dim=128, noise_stage_num = 1)
        state_dict = torch.load(G_model_path)
        netG.load_state_dict(state_dict)
        noise = Variable(torch.FloatTensor(batch_size, 128))
        bgnoise = Variable(torch.FloatTensor(batch_size, 128))

        noise = noise.cuda()
        bgnoise = bgnoise.cuda()
        netG = netG.cuda()
        text_encoder = text_encoder.cuda()

        netG.eval()
        text_encoder.eval()
        
        saved_cnt = 0

        #temp = ""
        tree_mu  = []
        tree_var = []
        water_mu = []
        water_var = []
        water_bird_set = [51, 53]
        tree_bird_set = [180, 166, 165, 163, 156, 98]


        print('Start evaluation process ...')
        for step, data in enumerate(data_loader, 0):
            _, ori_imgs, text, text_len,  _, _, filename, sentence_id, cls_id = FineGAN_evaluator.__prepare_data(data, gpu)
            #for i in range(len(filename)):
            #    temp += str(filename[i]) + str(sentence_id[i]) + str(text_len[i]) + "\n"

            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(text, text_len, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (text == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            mask = mask.cuda()
            
            noise.data.normal_(0, 1)
            bgnoise.data.normal_(0, 1)
            with torch.set_grad_enabled(False):
                bg_image, fake_images, ob_masks, mu, log_var, obs = netG(noise, sent_emb, words_embs, mask, bgnoise)
                #bg_image, fake_images, ob_masks, mu, log_var, obs = netG(noise, sent_emb, words_embs, mask, bgnoise)

            for i in range(batch_size):
                if cls_id[i] in water_bird_set:
                    water_mu.append(mu[i].cpu().numpy())
                    water_var.append(log_var[i].cpu().numpy())
                # else:
                #     tree_mu.append(mu[i].cpu().numpy())
                #     tree_var.append(log_var[i].cpu().numpy())
                elif cls_id[i] in tree_bird_set:
                    tree_mu.append(mu[i].cpu().numpy())
                    tree_var.append(log_var[i].cpu().numpy())
            
            low_res_img = fake_images[0]
            low_res_img = low_res_img.detach().cpu().numpy()
            low_res_img = np.transpose(low_res_img, [0, 2, 3, 1])

            high_res_img = fake_images[-1]
            high_res_img = high_res_img.detach().cpu().numpy()
            high_res_img = np.transpose(high_res_img, [0, 2, 3, 1])

            ori_imgs = ori_imgs.detach().cpu().numpy()
            ori_imgs = np.transpose(ori_imgs, [0, 2, 3, 1])

            bg_image = bg_image.detach().cpu().numpy()
            bg_image = np.transpose(bg_image, [0, 2, 3, 1])

            for i in range(2):
                obs[i] = obs[i].detach().cpu().numpy()
                obs[i] = np.transpose(obs[i], [0, 2, 3, 1])

            for i in range(batch_size):
                # filename[i] should in format %folder_name%/%file_name%
                folder_name = filename[i][:filename[i].index('/')]
                file_name   = filename[i][filename[i].index('/')+1:] + '_s-{}.png'.format(sentence_id[i])

                folder_path = os.path.join(save_path, folder_name)
                file_path   = os.path.join(folder_path, file_name)

                if not os.path.isdir(folder_path):
                    print('Make a new folder: ', folder_path)
                    mkdir_p(folder_path)

                imsave(file_path, (FineGAN_evaluator.__normalize(high_res_img[i]) * 255.0).astype(np.uint8))
                saved_cnt += 1

                file_name = filename[i][filename[i].index('/')+1:] + '_s-{}.eps'.format(sentence_id[i])
                folder_path = os.path.join(save_path, folder_name)
                file_path = os.path.join(folder_path, file_name)

                imsave(file_path, (FineGAN_evaluator.__normalize(high_res_img[i]) * 255.0).astype(np.uint8))
                '''
                process_presentation_folder_name = \
                    filename[i][filename[i].index('/')+1:] + '_s-{}'.format(sentence_id[i])
                process_presentation_folder_path = os.path.join(folder_path, process_presentation_folder_name)

                if not os.path.isdir(process_presentation_folder_path):
                    mkdir_p(process_presentation_folder_path)

                for j in range(len(ob_masks)):
                    ob_mask_saved_name_png = filename[i][filename[i].index('/')+1:] + \
                        '_s-{}_obmask{}.png'.format(sentence_id[i], j)
                    ob_mask_saved_name_eps = filename[i][filename[i].index('/')+1:] + \
                        '_s-{}_obmask{}.eps'.format(sentence_id[i], j)
                    ob_mask_saved_path_png = os.path.join(process_presentation_folder_path, ob_mask_saved_name_png)
                    ob_mask_saved_path_eps = os.path.join(process_presentation_folder_path, ob_mask_saved_name_eps)
                    imsave(ob_mask_saved_path_png, FineGAN_evaluator.__normalize(np.squeeze(ob_masks[j][i].detach().cpu().numpy())) * 255.0)
                    imsave(ob_mask_saved_path_eps, FineGAN_evaluator.__normalize(np.squeeze(ob_masks[j][i].detach().cpu().numpy())) * 255.0)
                
                low_file_name_png = filename[i][filename[i].index('/')+1:] + '_64s-{}.png'.format(sentence_id[i])
                low_file_name_eps = filename[i][filename[i].index('/')+1:] + '_64s-{}.eps'.format(sentence_id[i])
                low_file_path_png = os.path.join(process_presentation_folder_path, low_file_name_png)
                low_file_path_eps = os.path.join(process_presentation_folder_path, low_file_name_eps)
                imsave(low_file_path_png, (FineGAN_evaluator.__normalize(low_res_img[i]) * 255.0).astype(np.uint8))
                imsave(low_file_path_eps, (FineGAN_evaluator.__normalize(low_res_img[i]) * 255.0).astype(np.uint8))
                
                file_name_png = filename[i][filename[i].index('/')+1:] + '_s-{}.png'.format(sentence_id[i])
                file_name_eps = filename[i][filename[i].index('/')+1:] + '_s-{}.eps'.format(sentence_id[i])
                file_path_png = os.path.join(process_presentation_folder_path, file_name_png)
                file_path_eps = os.path.join(process_presentation_folder_path, file_name_eps)
                imsave(file_path_png, (FineGAN_evaluator.__normalize(high_res_img[i]) * 255.0).astype(np.uint8))
                imsave(file_path_eps, (FineGAN_evaluator.__normalize(high_res_img[i]) * 255.0).astype(np.uint8))

                bg_name_png = filename[i][filename[i].index('/')+1:] + 'bg_s-{}.png'.format(sentence_id[i])
                bg_name_eps = filename[i][filename[i].index('/')+1:] + 'bg_s-{}.eps'.format(sentence_id[i])
                bg_path_png = os.path.join(process_presentation_folder_path, bg_name_png)
                bg_path_eps = os.path.join(process_presentation_folder_path, bg_name_eps)
                imsave(bg_path_png, (FineGAN_evaluator.__normalize(bg_image[i]) * 255.0).astype(np.uint8))
                imsave(bg_path_eps, (FineGAN_evaluator.__normalize(bg_image[i]) * 255.0).astype(np.uint8))

                ob64_name_png = filename[i][filename[i].index('/')+1:] + 'ob64_s-{}.png'.format(sentence_id[i])
                ob64_name_eps = filename[i][filename[i].index('/')+1:] + 'ob64_s-{}.eps'.format(sentence_id[i])
                ob64_path_png = os.path.join(process_presentation_folder_path, ob64_name_png)
                ob64_path_eps = os.path.join(process_presentation_folder_path, ob64_name_eps)
                imsave(ob64_path_png, (FineGAN_evaluator.__normalize(obs[0][i]) * 255.0).astype(np.uint8))
                imsave(ob64_path_eps, (FineGAN_evaluator.__normalize(obs[0][i]) * 255.0).astype(np.uint8))

                ob128_name_png = filename[i][filename[i].index('/')+1:] + 'ob128_s-{}.png'.format(sentence_id[i])
                ob128_name_eps = filename[i][filename[i].index('/')+1:] + 'ob128_s-{}.eps'.format(sentence_id[i])
                ob128_path_png = os.path.join(process_presentation_folder_path, ob128_name_png)
                ob128_path_eps = os.path.join(process_presentation_folder_path, ob128_name_eps)
                imsave(ob128_path_png, (FineGAN_evaluator.__normalize(obs[1][i]) * 255.0).astype(np.uint8))
                imsave(ob128_path_eps, (FineGAN_evaluator.__normalize(obs[1][i]) * 255.0).astype(np.uint8))

                origin_file_name_png = 'origin.png'
                origin_file_name_eps = 'origin.eps'
                origin_file_path_png = os.path.join(process_presentation_folder_path, origin_file_name_png)
                origin_file_path_eps = os.path.join(process_presentation_folder_path, origin_file_name_eps)
                imsave(origin_file_path_png, (FineGAN_evaluator.__normalize(ori_imgs[i]) * 255.0).astype(np.uint8))
                imsave(origin_file_path_eps, (FineGAN_evaluator.__normalize(ori_imgs[i]) * 255.0).astype(np.uint8))

                text_path = os.path.join(process_presentation_folder_path, 'object_description.txt')
                text_log = ""
                ttext = text.detach().cpu().numpy()
                for j in range(text_len[i]):
                    text_log = text_log + word_dict[ttext[i][j]] + ' '

                with open(text_path, 'w') as f:
                    f.write(text_log)
                '''

        tree_mu = np.array(tree_mu)
        tree_var = np.array(tree_var)
        water_mu = np.array(water_mu)
        water_var = np.array(water_var)
        print(tree_mu.size, tree_var.size, water_mu.size, water_var.size)

        np.save('tree_mu.npy', np.array(tree_mu))
        np.save('tree_var.npy', np.array(tree_var))
        np.save('water_mu.npy', np.array(water_mu))
        np.save('water_var.npy', np.array(water_var))
        print('Evaluation finished! {} images saved totally'.format(saved_cnt))

class FineGAN_evaluator_by_latent(object):

    @staticmethod
    def __prepare_data(data, gpu=0):
        full_image, imgs, text, text_len, imgsx64, object_mask, key, sentence_id = data
        # sort data by the length in a decreasing order
        text_len, sorted_text_indice = torch.sort(text_len, 0, True)

        imgs = imgs[sorted_text_indice]
        imgsx64 = imgsx64[sorted_text_indice]
        full_image = full_image[sorted_text_indice]
        object_mask = object_mask[sorted_text_indice]
        text = text[sorted_text_indice].squeeze()
        sorted_key = [key[i] for i in sorted_text_indice.numpy()]
        sentence_id = [sentence_id[i] for i in sorted_text_indice.numpy()]

        # send data to device
        text = text.cuda()
        imgs = imgs.cuda()
        imgsx64 = imgsx64.cuda()
        text_len = text_len.cuda()
        full_image = full_image.cuda()
        object_mask = object_mask.cuda()

        return full_image, imgs, text, text_len, imgsx64, object_mask, sorted_key, sentence_id

    @staticmethod
    def __normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def evaluate(text_encoder, data_loader, save_path, G_model_path, word_dict, latent, latent_len, words_emb, gpu=0):
        torch.cuda.set_device(gpu)
        netG = Generator(t_dim=256, s_dim=256, noise_stage_num=1)
        state_dict = torch.load(G_model_path)
        netG.load_state_dict(state_dict)

        netG = netG.cuda()
        text_encoder = text_encoder.cuda()

        netG.eval()
        text_encoder.eval()

        saved_cnt = 0

        mask = torch.zeros((1, 18), dtype=torch.bool).cuda()

        with torch.set_grad_enabled(False):
            bg_image, fake_images, ob_masks, NI_obs, NI_masks, obs, bgs, mu, log_var = netG(1, words_emb, mask, latent)

        high_res_img = fake_images[0]
        high_res_img = high_res_img.detach().cpu().numpy()
        high_res_img = np.transpose(high_res_img, [0, 2, 3, 1])

        imsave(save_path, (FineGAN_evaluator_by_latent.__normalize(high_res_img[0]) * 255.0).astype(np.uint8))
        saved_cnt += 1

        print('Evaluation finished! {} images saved totally'.format(saved_cnt))
