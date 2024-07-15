import numpy as np
import torch

from models.asp_loss import AdaptiveSupervisedPatchNCELoss
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .gauss_pyramid import Gauss_Pyramid_Conv
import util.util as util
from .SRC import SRC_Loss
from ipdb import set_trace
from models.reg.reg import Reg
from models.reg.transformer import Transformer_2D
from .ssp_loss import SSPLoss
import time
class BSPModel(BaseModel):
    """ Contrastive Paired Translation (CPT).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0)  # no image pooling

        # FDL:
        parser.add_argument('--lambda_gp', type=float, default=1.0, help='weight for Gaussian Pyramid reconstruction loss')
        parser.add_argument('--gp_weights', type=str, default='uniform', help='weights for reconstruction pyramids.')
        parser.add_argument('--lambda_asp', type=float, default=0.0, help='weight for ASP loss')
        parser.add_argument('--asp_loss_mode', type=str, default='none', help='"scheduler_lookup" options for the ASP loss. Options for both are listed in Fig. 3 of the paper.')
        parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsample in G')

        #SRC:
        parser.add_argument('--lambda_src', type=float, default=0.1, help='weight for SRC loss: SRC(G(X), X)')
        parser.add_argument('--nce_idt_src', type=util.str2bool, nargs='?', const=True, default=False, help='use SRC loss for identity mapping: NCE(G(X), Y))')
        # BSP:
        parser.add_argument('--lambda_bsp', type=float, default=0.0, help='weight for BSP loss')
        parser.add_argument('--bsp_layers_patches', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')

        # REG:
        parser.add_argument('--reg_mode', type=bool, default=False, help='if use Reg')
        parser.add_argument('--lambda_smooth', type=float, default=0.0, help='compute smooth loss on reg')
        # SSP:
        parser.add_argument('--lambda_ssp', type=float, default=0.0, help='weight for SSP loss')
        parser.add_argument('--ssp_patches', type=int, default=1, help='numbers for selected patches')
        parser.add_argument('--ssp_out_sum', type=util.str2bool, nargs='?', const=True, default=False, help='ssp loss out sum')


        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=False,
                n_epochs=20, n_epochs_decay=10
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.bsp_layers_patches = [int(i) for i in self.opt.bsp_layers_patches.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']
        
        
        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
            if self.opt.reg_mode is True:
                self.netR = Reg(opt.crop_size, opt.crop_size, 3, 3).to(self.device)
                self.optimizer_reg = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.model_names += ['R']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.lambda_src > 0:
                self.criterionSRC = SRC_Loss(self.opt).to(self.device)
                self.loss_names += ['SRC']
                if self.opt.nce_idt_src:
                    self.loss_names += ['SRC_idt']

            if self.opt.lambda_bsp > 0:
                # self.criterionASP = AdaptiveSupervisedPatchNCELoss(self.opt).to(self.device)
                self.loss_names += ['BSP']
            
            if self.opt.lambda_ssp > 0:
                # self.criterionASP = AdaptiveSupervisedPatchNCELoss(self.opt).to(self.device)
                self.loss_names += ['SSP']
                self.criterionSSP = SSPLoss(self.opt).to(self.device)

            if self.opt.reg_mode is True:
                
                self.spatial_transform = Transformer_2D().to(self.device)
                self.loss_names += ['reg_smooth']
                self.loss_names += ['reg_nce']

            if self.opt.lambda_gp > 0:
                self.P = Gauss_Pyramid_Conv(num_high=5)
                self.criterionGP = torch.nn.L1Loss().to(self.device)
                if self.opt.gp_weights == 'uniform':
                    self.gp_weights = [1.0] * 6
                else:
                    self.gp_weights = eval(self.opt.gp_weights)
                self.loss_names += ['GP']

            if self.opt.lambda_asp > 0:
                self.criterionASP = AdaptiveSupervisedPatchNCELoss(self.opt).to(self.device)
                self.loss_names += ['ASP']


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0 or self.opt.lambda_asp > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        if self.opt.reg_mode is True:
            self.optimizer_reg.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.reg_mode is True:
            self.optimizer_reg.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_epoch(self, epoch):
        self.train_epoch = epoch

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']

    def forward(self):
        # self.netG.print()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B

        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
        if self.opt.nce_idt:
            feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        loss_NCE_all = self.loss_NCE

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE_Y = 0.0
        loss_NCE_all += self.loss_NCE_Y

        # FDL: NCE between the noisy pairs (fake_B and real_B)
        if self.opt.lambda_asp > 0:
            self.loss_ASP = self.calculate_NCE_loss(feat_real_B, feat_fake_B, self.netF, self.nce_layers, paired=True)
        else:
            self.loss_ASP = 0.0
        loss_NCE_all += self.loss_ASP
        # BSPloss
        if self.opt.lambda_bsp > 0:
            self.loss_BSP = self.calculate_BSP_loss(feat_real_B, feat_fake_B, self.netF, self.nce_layers)
        else:
            self.loss_BSP = 0.0

        # SSPloss
        if self.opt.lambda_ssp > 0:
            
            self.loss_SSP = self.calculate_SSP_loss(feat_real_B, feat_fake_B, self.netF, self.nce_layers, self.opt.ssp_patches)
        else:
            self.loss_SSP = 0.0
        loss_NCE_all += self.loss_SSP
        # reg mode
        if self.opt.reg_mode is True:
            # set_trace()
            Trans = self.netR(self.fake_B,self.real_B) 
            SysRegist_A2B = self.spatial_transform(self.fake_B,Trans)
            feat_fake_B_regist = self.netG(SysRegist_A2B, self.nce_layers, encode_only=True)
            self.loss_reg_nce = self.calculate_NCE_loss(feat_real_B, feat_fake_B_regist, self.netF, self.nce_layers, paired=False)
            # SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
            
            ####smooth loss
            self.loss_reg_smooth = self.opt.lambda_smooth * self.smooothing_loss(Trans)
            # set_trace()
        else:
            self.loss_reg_nce = 0.0
            self.loss_reg_smooth = 0.0
        loss_NCE_all += self.loss_reg_nce + self.loss_reg_smooth

        if self.opt.lambda_src > 0:
            fake_B_pool, sample_ids = self.netF(feat_fake_B, self.opt.num_patches, None)
            real_A_pool, _ = self.netF(feat_real_A, self.opt.num_patches, sample_ids)
            self.loss_SRC, _ = self.calculate_R_loss(real_A_pool, fake_B_pool, epoch=self.train_epoch)
            self.loss_SRC_total = self.loss_SRC
            if self.opt.nce_idt_src:

                real_B_pool, _ = self.netF(feat_real_B, self.opt.num_patches, sample_ids)
                self.loss_SRC_idt, _ = self.calculate_R_loss(real_B_pool, fake_B_pool, epoch=self.train_epoch)

                self.loss_SRC_total = (self.loss_SRC + self.loss_SRC_idt) / 2.0
            else:
                self.loss_SRC_idt = 0.0
        else:
            self.loss_SRC = 0.0
            self.loss_SRC_idt = 0.0
            self.loss_SRC_total = 0.0

        loss_NCE_all += self.loss_SRC_total

        # FDL: compute loss on Gaussian pyramids
        if self.opt.lambda_gp > 0:
            p_fake_B = self.P(self.fake_B)
            p_real_B = self.P(self.real_B)
            loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake_B, p_real_B)]
            weights = self.gp_weights
            loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
            self.loss_GP = torch.mean(torch.stack(loss_pyramid)) * self.opt.lambda_gp
        else:
            self.loss_GP = 0

        self.loss_G = self.loss_G_GAN + loss_NCE_all + self.loss_GP
        return self.loss_G

    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, nce_layers, paired=False):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)
        # set_trace()
        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            if paired:
                loss = self.criterionASP(f_q, f_k, self.current_epoch) * self.opt.lambda_asp
            else:
                loss = self.criterionNCE(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def calculate_BSP_loss(self, feat_src, feat_tgt, netF, nce_layers):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, -1, None)
        feat_q_pool, _ = netF(feat_q, -1, sample_ids)

        total_nce_loss = 0.0
        for index, (f_q, f_k) in enumerate(zip(feat_q_pool, feat_k_pool)):
            inner_products = (f_q * f_k).sum(dim=1)
            _, top_m_indices = torch.topk(inner_products, self.bsp_layers_patches[index])
            f_q_new = f_q[top_m_indices]
            f_k_new = f_k[top_m_indices]
            # loss = self.criterionASP(f_q, f_k, self.current_epoch) * self.opt.lambda_asp
            
            loss = self.criterionNCE(f_q_new, f_k_new) * self.opt.lambda_bsp
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_SSP_loss(self, feat_src, feat_tgt, netF, nce_layers, patches=1):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        start_time = time.time()
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _, sample_ids_surrund = netF(feat_q, self.opt.num_patches, sample_ids, return_surrund=True)
        # set_trace()
        # print("id_surrund_time: ", time.time() - start_time)
        feat_k_pool_all, _ = netF(feat_k, -1, None)
        # print("compute total_time: ", time.time() - start_time)
        # feat_k_pool_all = feat_k_pool_all.detach()
        
        total_nce_loss = 0.0
        for surrund_id, f_q, f_k, f_k_all in zip(sample_ids_surrund, feat_q_pool, feat_k_pool, feat_k_pool_all):
            
            start_time = time.time()
            # f_k_all = f_k_all.detach()
            surrund_id = torch.tensor(surrund_id, dtype=torch.long, device=f_q.device)
            mask = (surrund_id == -1) # 256 * 8
            surrund_id[mask] = 0 # 256 * 8
            patches_num, surrund_num = surrund_id.shape   
            dim = f_k_all.shape[-1]
            surrund_id_expanded = surrund_id.unsqueeze(-1).expand(-1, -1, dim) # 256 * 8 * 256
            f_k_all_expanded = f_k_all.unsqueeze(0).expand(patches_num, -1, -1) # patches_num * (H * W) * 256
            f_surrund_id = torch.gather(f_k_all_expanded, 1, surrund_id_expanded) # 256 * 8 * 256
            temp_f_q = f_q.unsqueeze(1)
            dot_logits = torch.bmm(temp_f_q, f_surrund_id.transpose(1, 2))
            dot_logits = dot_logits.squeeze(1)
            dot_logits[mask] = -10000

            _, topk_indices = torch.topk(dot_logits, k=self.opt.ssp_patches, dim=1) # 256 * 3
            
            
            # topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, dim)
            # comp_f = torch.gather(f_surrund_id, 1, topk_indices_expanded)
            batch_indices = torch.arange(f_surrund_id.shape[0]).unsqueeze(1).expand(-1, topk_indices.shape[1])
            comp_f = f_surrund_id[batch_indices, topk_indices]
                # f_q = f_q.unsqueeze(1)

            # print("compute comp_f time: ", time.time() - start_time)
            # set_trace()
            loss = self.criterionSSP(f_q, f_k, comp_f, out=self.opt.ssp_out_sum) * self.opt.lambda_ssp
            total_nce_loss += loss.mean()
            # print("compute loss time: ", time.time() - start_time)
        return total_nce_loss / n_layers
    
    def calculate_R_loss(self, src, tgt, only_weight=False, epoch=None):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_SRC_loss = 0.0
        weights=[]
        for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, self.nce_layers):
            loss_SRC, weight = self.criterionSRC(f_q, f_k, only_weight, epoch)
            total_SRC_loss += loss_SRC * self.opt.lambda_src
            weights.append(weight)
        return total_SRC_loss / n_layers, weights

    def smooothing_loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        dx = dx*dx
        dy = dy*dy
        d = torch.mean(dx) + torch.mean(dy)
        grad = d 
        return d