from torch.nn import CrossEntropyLoss, BCELoss, L1Loss, Tanh
from torch.nn.modules import loss
from utils.get_optimizer import get_optimizer
from utils.TripletLoss import TripletLoss
import torch
from torch.distributions import normal
import numpy as np
import copy
from utils.tensor2img import tensor2im
from opt import opt

from torchvision import transforms
from PIL import Image

class Loss(loss._Loss):
    def __init__(self, model):
        super(Loss, self).__init__()
        self.batch_size = opt.batchid * opt.batchimage
        self.num_gran = 8
        self.tanh = Tanh()
        self.l1_loss = L1Loss()
        self.bce_loss = BCELoss()
        self.cross_entropy_loss = CrossEntropyLoss()

        self.model = model
        self.optimizer, self.optimizer_D, self.optimizer_DC = get_optimizer(model)

    def get_positive_pairs(self):
        idx = []
        for i in range(self.batch_size):
            r = i
            while r == i:
                r = int(torch.randint(
                    low=opt.batchimage * (i // opt.batchimage), high=opt.batchimage * (i // opt.batchimage + 1),
                    size=(1,)).item())
            idx.append(r)
        return idx

    def region_wise_shuffle(self, id, ps_idx):
        sep_id = id.clone()
        idx = torch.tensor([0] * (self.num_gran))
        while (torch.sum(idx) == 0) and (torch.sum(idx) == self.num_gran):
            idx = torch.randint(high=2, size=(self.num_gran,))

        for i in range(self.num_gran):
            if idx[i]:
                sep_id[:, opt.feat_id * i:opt.feat_id * (i + 1)] = id[ps_idx][:, opt.feat_id * i:opt.feat_id * (i + 1)]
        return sep_id

    def get_noise(self):
        return torch.randn(self.batch_size, opt.feat_niz, device=opt.device)

    def make_onehot(self, label):
        onehot_vec = torch.zeros(self.batch_size, opt.num_cls)
        for i in range(label.size()[0]):
            onehot_vec[i, label[i]] = 1
        return onehot_vec

    def set_parameter(self, m, train=True):
        if train:
            for param in m.parameters():
                param.requires_grad = True
            m.apply(self.set_bn_to_train)
        else:
            for param in m.parameters():
                param.requires_grad = False
            m.apply(self.set_bn_to_eval)

    def set_bn_to_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()

    def set_bn_to_train(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.train()

    def set_model(self, batch=None):
        self.model.C.zero_grad()
        self.model.G.zero_grad()
        self.model.D.zero_grad()

        if opt.stage == 0:
            self.set_parameter(self.model.C, train=True)
            self.set_parameter(self.model.G, train=False)
            self.set_parameter(self.model.D, train=False)
            self.set_parameter(self.model.DC, train=False)

        elif opt.stage == 1:
            self.set_parameter(self.model.C, train=False)
            cloth_dict1 = self.model.C.get_modules(self.model.C.cloth_dict1())
            cloth_dict2 = self.model.C.get_modules(self.model.C.cloth_dict2())
            for i in range(np.shape(cloth_dict1)[0]):
                self.set_parameter(cloth_dict1[i], train=True)
            for i in range(np.shape(cloth_dict2)[0]):
                self.set_parameter(cloth_dict2[i], train=True)
            self.set_parameter(self.model.G, train=False)
            self.set_parameter(self.model.D, train=False)
            self.set_parameter(self.model.DC, train=False)

        elif opt.stage == 2:
            self.set_parameter(self.model.C, train=False)
            nid_dict1 = self.model.C.get_modules(self.model.C.nid_dict1())
            nid_dict2 = self.model.C.get_modules(self.model.C.nid_dict2())
            # cloth_dict1 = self.model.C.get_modules(self.model.C.cloth_dict1())
            # cloth_dict2 = self.model.C.get_modules(self.model.C.cloth_dict2())
            for i in range(np.shape(nid_dict1)[0]):
                self.set_parameter(nid_dict1[i], train=True)
            for i in range(np.shape(nid_dict2)[0]):
                self.set_parameter(nid_dict2[i], train=True)
            # for i in range(np.shape(cloth_dict1)[0]):
                # self.set_parameter(cloth_dict1[i], train=True)
            # for i in range(np.shape(cloth_dict2)[0]):
                # self.set_parameter(cloth_dict2[i], train=True)
            self.set_parameter(self.model.G, train=True)
            self.set_parameter(self.model.D, train=True)
            self.set_parameter(self.model.DC, train=True)



    def id_related_loss(self, labels, outputs):
        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[1:1 + self.num_gran]]
        return sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

    def cloth_related_loss(self, labels, outputs):
        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[-6]]
        return sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

    def rgb_gray_distance(self, rgb_outputs, gray_outputs):
        return torch.sum(torch.pairwise_distance(rgb_outputs[0], gray_outputs[0]))

    def KL_loss(self, outputs):
        list_mu = outputs[-3]
        list_lv = outputs[-2]
        loss_KL = 0.
        for i in range(np.size(list_mu)):
            loss_KL += torch.sum(0.5 * (list_mu[i] ** 2 + torch.exp(list_lv[i]) - list_lv[i] - 1))
        return loss_KL / np.size(list_mu)



    def GAN_loss(self, inputs, outputs, labels, cloth_labels, epoch):
        id = outputs[0]
        nid = outputs[-1]
        cloth = outputs[-4]
        one_hot_labels = self.make_onehot(labels).to(opt.device)

        # Auto Encoder
        auto_G_in = torch.cat((id, cloth, nid, self.get_noise()), dim=1)
        auto_G_out = self.model.G.forward(auto_G_in, one_hot_labels)

        # Positive Shuffle
        ps_idx = self.get_positive_pairs()
        ps_G_in = torch.cat((id[ps_idx], cloth, nid, self.get_noise()), dim=1)
        ps_G_out = self.model.G.forward(ps_G_in, one_hot_labels)

        s_ps_G_in = torch.cat((id, cloth[ps_idx], nid, self.get_noise()), dim=1)
        s_ps_G_out = self.model.G.forward(s_ps_G_in, one_hot_labels)

        s2_ps_G_in = torch.cat((id[ps_idx], cloth[ps_idx], nid, self.get_noise()), dim=1)
        s2_ps_G_out = self.model.G.forward(s2_ps_G_in, one_hot_labels)

        # Negative Shuffle
        neg_idx = ps_idx[::-1]
        neg_G_in = torch.cat((id, cloth[neg_idx], nid[neg_idx], self.get_noise()), dim=1)
        neg_G_out = self.model.G.forward(neg_G_in, one_hot_labels)

        ############################################## D_loss ############################################
        D_real = self.model.D(inputs)
        REAL_LABEL = torch.FloatTensor(D_real.size()).uniform_(0.7, 1.0).to(opt.device)
        D_real_loss = self.bce_loss(D_real, REAL_LABEL)

        auto_D_fake = self.model.D(auto_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(auto_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        auto_D_fake_loss = self.bce_loss(auto_D_fake, FAKE_LABEL)

        ps_D_fake = self.model.D(ps_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(ps_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        ps_D_fake_loss = self.bce_loss(ps_D_fake, FAKE_LABEL)

        s_ps_D_fake = self.model.D(s_ps_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(s_ps_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        s_ps_D_fake_loss = self.bce_loss(s_ps_D_fake, FAKE_LABEL)

        s2_ps_D_fake = self.model.D(s2_ps_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(s2_ps_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        s2_ps_D_fake_loss = self.bce_loss(s2_ps_D_fake, FAKE_LABEL)

        neg_D_fake = self.model.D(neg_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(neg_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        neg_D_fake_loss = self.bce_loss(neg_D_fake, FAKE_LABEL)

        D_loss = (D_real_loss + auto_D_fake_loss + ps_D_fake_loss + s_ps_D_fake_loss + s2_ps_D_fake_loss + neg_D_fake_loss)
        D_loss.backward()
        self.optimizer_D.step()

        ############################################## G_loss ##############################################
        auto_D_fake = self.model.D(auto_G_out)
        auto_I_fake, auto_C_fake = self.model.DC(auto_G_out)
        REAL_LABEL = torch.ones_like(auto_D_fake)
        auto_D_fake_loss = self.bce_loss(auto_D_fake, REAL_LABEL)
        auto_I_fake_loss = self.cross_entropy_loss(auto_I_fake, labels)
        auto_C_fake_loss = self.cross_entropy_loss(auto_C_fake, cloth_labels)
        # auto_cls_loss = self.cross_entropy_loss(auto_G_cls, labels)

        ps_D_fake = self.model.D(ps_G_out)
        ps_I_fake, ps_C_fake = self.model.DC(ps_G_out)
        REAL_LABEL = torch.ones_like(ps_D_fake)
        ps_D_fake_loss = self.bce_loss(ps_D_fake, REAL_LABEL)
        ps_I_fake_loss = self.cross_entropy_loss(ps_I_fake, labels)
        ps_C_fake_loss = self.cross_entropy_loss(ps_C_fake, cloth_labels)
        # ps_cls_loss = self.cross_entropy_loss(ps_G_cls, labels)

        s_ps_D_fake = self.model.D(s_ps_G_out)
        s_ps_I_fake, s_ps_C_fake = self.model.DC(s_ps_G_out)
        REAL_LABEL = torch.ones_like(s_ps_D_fake)
        s_ps_D_fake_loss = self.bce_loss(s_ps_D_fake, REAL_LABEL)
        s_ps_I_fake_loss = self.cross_entropy_loss(s_ps_I_fake, labels)
        s_ps_C_fake_loss = self.cross_entropy_loss(s_ps_C_fake, cloth_labels[ps_idx])

        s2_ps_D_fake = self.model.D(s2_ps_G_out)
        s2_ps_I_fake, s2_ps_C_fake = self.model.DC(s2_ps_G_out)
        REAL_LABEL = torch.ones_like(s2_ps_D_fake)
        s2_ps_D_fake_loss = self.bce_loss(s2_ps_D_fake, REAL_LABEL)
        s2_ps_I_fake_loss = self.cross_entropy_loss(s2_ps_I_fake, labels)
        s2_ps_C_fake_loss = self.cross_entropy_loss(s2_ps_C_fake, cloth_labels[ps_idx])

        neg_D_fake  = self.model.D(neg_G_out)
        neg_I_fake, neg_C_fake = self.model.DC(neg_G_out)
        REAL_LABEL = torch.ones_like(neg_D_fake)
        neg_D_fake_loss = self.bce_loss(neg_D_fake, REAL_LABEL)
        neg_I_fake_loss = self.cross_entropy_loss(neg_I_fake, labels)
        neg_C_fake_loss = self.cross_entropy_loss(neg_C_fake, cloth_labels[neg_idx])

        auto_imgr_loss = self.l1_loss(auto_G_out, self.tanh(inputs))
        ps_imgr_loss = self.l1_loss(ps_G_out, self.tanh(inputs))
        s_ps_imgr_loss = self.l1_loss(s_ps_G_out, s2_ps_G_out)

        '''
        img = tensor2im(auto_G_out[0])
        img = Image.fromarray(img)
        img.save('test_image' + '/aaa' + '.jpg')
        img = tensor2im(inputs[0])
        img = Image.fromarray(img)
        img.save('test_image' + '/anchor' + '.jpg')
        '''

        if epoch > 50:
            G_loss = (auto_D_fake_loss + ps_D_fake_loss + s_ps_D_fake_loss + s2_ps_D_fake_loss + neg_D_fake_loss) + \
                 (auto_C_fake_loss + ps_C_fake_loss + s_ps_C_fake_loss + s2_ps_C_fake_loss + neg_C_fake_loss) * 2 + \
                 (auto_I_fake_loss + ps_I_fake_loss + s_ps_I_fake_loss + s2_ps_I_fake_loss + neg_I_fake_loss) * 2 + \
                 (auto_imgr_loss + ps_imgr_loss + s_ps_imgr_loss) * 10
        else:
            G_loss = (auto_D_fake_loss + ps_D_fake_loss + s_ps_D_fake_loss + s2_ps_D_fake_loss + neg_D_fake_loss) + \
                     (auto_imgr_loss + ps_imgr_loss + s_ps_imgr_loss) * 10

        ############################################################################################
        return D_loss, G_loss

    def forward(self, rgb, labels, cloth_labels, batch, epoch):
        self.set_model(batch)
        rgb_outputs = self.model.C(rgb)

        if opt.stage == 0:
            Rgb_CE = self.id_related_loss(labels, rgb_outputs)
            IDcnt = 0
            IDtotal = opt.batchid * opt.batchimage * self.num_gran
            for classifyprobabilities in rgb_outputs[1:1 + self.num_gran]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    label = labels[i]
                    if class_ == label:
                        IDcnt += 1

            loss_sum = Rgb_CE

            print('\rRgb_CE:%.2f' % (
                Rgb_CE.data.cpu().numpy()
            ), end=' ')
            return loss_sum, [Rgb_CE.data.cpu().numpy()], [[IDcnt, IDtotal], [1, 1]]

        elif opt.stage == 1:
            # D_outputs_id, D_outputs_cloth = self.model.DC(rgb)

            Cloth_CE = self.cloth_related_loss(cloth_labels, rgb_outputs)
            Clothcnt = 0
            Clothtotal = opt.batchid * opt.batchimage * 3
            for classifyprobabilities in rgb_outputs[-6]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    cloth_label = cloth_labels[i]
                    if class_ == cloth_label:
                        Clothcnt += 1

            # D_I = self.cross_entropy_loss(D_outputs_id, labels)
            # D_C = self.cross_entropy_loss(D_outputs_cloth, cloth_labels)
            # DC_loss = D_I + D_C
            # DC_loss.backward()
            # self.optimizer_DC.step()
            loss_sum = Cloth_CE

            print('\rCloth_CE:%.2f' % (
                Cloth_CE.data.cpu().numpy()
                ), end=' ')
            return loss_sum, \
                   [Cloth_CE.data.cpu().numpy()], \
                   [[1, 1], [Clothcnt, Clothtotal]]

        elif opt.stage == 2:
            D_outputs_id, D_outputs_cloth = self.model.DC(rgb)
            D_I = self.cross_entropy_loss(D_outputs_id, labels)
            D_C = self.cross_entropy_loss(D_outputs_cloth, cloth_labels)
            DC_loss = D_I + D_C
            DC_loss.backward()
            self.optimizer_DC.step()

            D_loss, G_loss = self.GAN_loss(rgb, rgb_outputs, labels, cloth_labels, epoch)
            KL_loss = self.KL_loss(rgb_outputs)

            loss_sum = G_loss + KL_loss / 500

            print('\rD_loss:%.2f  G_loss:%.2f KL:%.2f D_I:%.2f D_C:%.2f' % (
                D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(),
                KL_loss.data.cpu().numpy(),
                D_I.data.cpu().numpy(),
                D_C.data.cpu().numpy()), end=' ')
            return loss_sum, \
                   [D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy(), KL_loss.data.cpu().numpy(), D_I.data.cpu().numpy(), D_C.data.cpu().numpy()], \
                   [[1, 1], [1, 1]]

        elif opt.stage == 3:

            D_outputs_id, D_outputs_cloth = self.model.DC(rgb)
            D_I = self.cross_entropy_loss(D_outputs_id, labels)
            D_C = self.cross_entropy_loss(D_outputs_cloth, cloth_labels)
            DC_loss = D_I + D_C
            DC_loss.backward()
            self.optimizer_DC.step()

            Rgb_CE = self.id_related_loss(labels, rgb_outputs)
            IDcnt = 0
            IDtotal = opt.batchid * opt.batchimage * self.num_gran
            for classifyprobabilities in rgb_outputs[1:1 + self.num_gran]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    label = labels[i]
                    if class_ == label:
                        IDcnt += 1

            cloth_loss = self.cloth_related_loss(cloth_labels, rgb_outputs)
            Clothcnt = 0
            Clothtotal = opt.batchid * opt.batchimage * 3
            for classifyprobabilities in rgb_outputs[-6]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    cloth_label = cloth_labels[i]
                    if class_ == cloth_label:
                        Clothcnt += 1

            D_loss, G_loss = self.GAN_loss(rgb, rgb_outputs, labels, cloth_labels, epoch)
            KL_loss = self.KL_loss(rgb_outputs)

            loss_sum = (Rgb_CE) * 20 + cloth_loss * 10 + G_loss / 2 + KL_loss / 100

            print('\rRgb_CE:%.2f Cloth:%.2f  D_loss:%.2f  G_loss:%.2f D_I:%.2f D_C:%.2f '
                  ' KL:%.2f' % (
                      Rgb_CE.data.cpu().numpy(),
                      cloth_loss.data.cpu().numpy(),
                      D_loss.data.cpu().numpy(),
                      G_loss.data.cpu().numpy(),
                      D_I.data.cpu().numpy(),
                      D_C.data.cpu().numpy(),
                      KL_loss.data.cpu().numpy()), end=' ')
            return loss_sum, \
                   [Rgb_CE.data.cpu().numpy(), cloth_loss.data.cpu().numpy(), D_loss.data.cpu().numpy(),
                    G_loss.data.cpu().numpy(), D_I.data.cpu().numpy(), D_C.data.cpu().numpy(),
                    KL_loss.data.cpu().numpy()], \
                   [[IDcnt, IDtotal], [Clothcnt, Clothtotal]]

        if opt.stage == 4:
            Rgb_CE = self.id_related_loss(labels, rgb_outputs)
            IDcnt = 0
            IDtotal = opt.batchid * opt.batchimage * self.num_gran
            for classifyprobabilities in rgb_outputs[1:1 + self.num_gran]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    label = labels[i]
                    if class_ == label:
                        IDcnt += 1

            loss_sum = Rgb_CE

            print('\rRgb_CE:%.2f' % (
                Rgb_CE.data.cpu().numpy()
            ), end=' ')
            return loss_sum, [Rgb_CE.data.cpu().numpy()], [[IDcnt, IDtotal], [1, 1]]
