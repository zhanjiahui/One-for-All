import torch
from bicubic import BicubicDownSample
import clip
import torch.nn.functional as F
from torchvision import transforms, utils
from lpips.lpips import LPIPS
from VGG16 import VGG16
from text_templates import imagenet_templates
# from stylegan2.model import Discriminator

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, loss_str, eps):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2] == ref_im.shape[3]
        im_size = ref_im.shape[2]  # 512
        factor = 1024 // im_size
        assert im_size * factor == 1024
        self.D_2 = BicubicDownSample(factor=2)
        self.D_4 = BicubicDownSample(factor=4)
        self.ref_im = ref_im
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.eps = eps
        self.l1_loss = torch.nn.L1Loss()

        clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')
        preprocess = transforms.Compose(#[transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                        #                                                    2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                        clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                        clip_preprocess.transforms[4:])  # + skip convert PIL to tensor
        self.preprocess = preprocess
        self.clip_model = clip_model


        # ref
        template_text_ref = compose_text_with_templates("dog", imagenet_templates)
        text_ref = clip.tokenize(template_text_ref).cuda()
        self.text_features_ref = self.clip_model.encode_text(text_ref)

        # gen
        template_text_tar = compose_text_with_templates("human face", imagenet_templates)
        text_tar = clip.tokenize(template_text_tar).cuda()
        self.text_features_tar = self.clip_model.encode_text(text_tar)

        self.ref_clip_latent = self.clip_model.encode_image(self.preprocess(self.D_2(self.ref_im)))

        self.gaussian_fit2_P = torch.load("gaussian_fit_stylegan2_P.pt")

        self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.cos2 = torch.nn.CosineSimilarity(dim=2)

        self.vgg = VGG16(requires_grad=False)

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if (isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        return ((gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).sum())

    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10 * ((gen_im_lr - ref_im).abs().mean((1, 2, 3)).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if (latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
            B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
            D = 2 * torch.atan2(A, B)
            D = ((D.pow(2) * 512).mean((1, 2)) / 8.).sum()
            return D

    def _loss_l1_clip(self, clip_latent, gen_im_lr, ref_im, **kwargs):
        return self.l1_loss(clip_latent, self.ref_clip_latent.unsqueeze(1))

    def _loss_l1_mlp_clip(self, clip_latent, gen_im_lr, ref_im, **kwargs):
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        # gen_clip_latent = gen_clip_latent / gen_clip_latent.clone().norm(dim=-1, keepdim=True)
        return self.l1_loss(clip_latent, gen_clip_latent.unsqueeze(1))

    def _loss_cos_mlp_clip(self, clip_latent, gen_im_lr, ref_im, **kwargs):
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        # gen_clip_latent = gen_clip_latent / gen_clip_latent.clone().norm(dim=-1, keepdim=True)
        return (1.0-self.cos(gen_clip_latent, clip_latent.squeeze(1)))[0]


    def _loss_l1_mlp_clip2(self, clip_latent, gen_im_lr, ref_im, **kwargs):
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        # gen_clip_latent = gen_clip_latent / gen_clip_latent.clone().norm(dim=-1, keepdim=True)
        return self.l1_loss(self.ref_clip_latent.unsqueeze(1), gen_clip_latent.unsqueeze(1))

    def _loss_clip_cos(self, clip_latent, gen_im_lr, ref_im, latent_avg, **kwargs):
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        return (1.0-self.cos(gen_clip_latent, self.ref_clip_latent))[0]

    def _loss_l1_mlp_clip2_style(self, clip_latent, gen_im_lr, ref_im, **kwargs):
        # print(clip_latent._version)
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        ref_clip_latent = self.ref_clip_latent - self.text_features_ref.detach()
        gen_clip_latent = gen_clip_latent - self.text_features_tar.detach()
        return self.l1_loss(ref_clip_latent.unsqueeze(1), gen_clip_latent.unsqueeze(1))

    def _loss_pca_clip_l1(self,gen_im_lr,ref_im,**kwargs):
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        ref_clip_latent = self.clip_model.encode_image(self.preprocess(ref_im))
        clusterable_gen = torch.mm(gen_clip_latent - self.pca_mean,torch.transpose(self.pca_compon[:300],dim0=1,dim1=0))
        clusterable_ref = torch.mm(ref_clip_latent - self.pca_mean,torch.transpose(self.pca_compon[:300],dim0=1,dim1=0))
        return self.l1_loss(clusterable_gen, clusterable_ref)

    def _loss_pca_clip_cos(self,gen_im_lr,ref_im,**kwargs):
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        ref_clip_latent = self.clip_model.encode_image(self.preprocess(ref_im))
        clusterable_gen = torch.mm(gen_clip_latent - self.pca_mean,
                                   torch.transpose(self.pca_compon[:300], dim0=1, dim1=0))
        clusterable_ref = torch.mm(ref_clip_latent - self.pca_mean,
                                   torch.transpose(self.pca_compon[:300], dim0=1, dim1=0))
        return (1.0 - self.cos(clusterable_gen, clusterable_ref))[0]

    def _loss_clip_cos_style(self, clip_latent, gen_im_lr, ref_im, latent_avg, **kwargs):
        gen_clip_latent = self.clip_model.encode_image(self.preprocess(gen_im_lr))
        ref_clip_latent = self.clip_model.encode_image(self.preprocess(ref_im))
        ref_clip_latent = ref_clip_latent - self.text_features_ref.detach()
        gen_clip_latent = gen_clip_latent - self.text_features_tar.detach()
        return (1.0-self.cos(gen_clip_latent, ref_clip_latent))[0]

    def _loss_w_norm(self, clip_latent, latent, gen_im_lr, ref_im, latent_avg, **kwargs):
        return torch.sum((latent - latent_avg).norm(2, dim=(1, 2))) / latent.shape[0]

    def _loss_lpips(self, clip_latent, latent, gen_im_lr, ref_im, latent_avg, **kwargs):
        return self.lpips_loss(gen_im_lr, ref_im)

    def _loss_ec(self, gen_im_lr, ref_im, **kwargs):
        return self.l1_loss(self.netEC(gen_im_lr),self.netEC(ref_im))

    def _loss_gram(self,gen_im_lr,ref_im,**kwargs):
        fake_feat3 = self.vgg(gen_im_lr)
        source_feat3 = self.vgg(ref_im)
        gram_fake3 = [gram_matrix(y) for y in fake_feat3]
        gram_source3 = [gram_matrix(y) for y in source_feat3]
        gram_loss = 0
        for fake, source in zip(gram_fake3, gram_source3):
            gram_loss = gram_loss + self.l1_loss(fake, source)
        return gram_loss

    def _loss_p_constraint(self,p_latent,searched_latent,p_lambda,**kwargs):
        return 100*p_lambda*self.l1_loss(p_latent, searched_latent*self.gaussian_fit2_P["std"]+self.gaussian_fit2_P["mean"])

    def forward(self, clip_latent, latent, gen_im, searched_latent=None, p_latent=None,p_lambda=None, latent_avg=None):
        var_dict ={
            'clip_latent': clip_latent,
            'latent': latent,
            'searched_latent': searched_latent,
            'p_latent': p_latent,
            'gen_im_lr': self.D_2(gen_im),
            'ref_im': self.ref_im,
            'latent_avg': latent_avg,
            'p_lambda': p_lambda
        }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'GEOCROSS': self._loss_geocross,
            'L1_CLIP': self._loss_l1_clip,
            'L1_MLP_CLIP': self._loss_l1_mlp_clip,
            'COS_MLP_CLIP': self._loss_cos_mlp_clip,
            'L1_MLP_CLIP2': self._loss_l1_mlp_clip2,
            'CLIP_COS': self._loss_clip_cos,
            'L1_MLP_CLIP2_STYLE': self._loss_l1_mlp_clip2_style,
            'PCA_CLIP_L1':self._loss_pca_clip_l1,
            'PCA_CLIP_COS':self._loss_pca_clip_cos,
            'CLIP_COS_STYLE': self._loss_clip_cos_style,
            # 'PARALLEL':self._loss_parallel,
            'W_NORM': self._loss_w_norm,
            'LPIPS': self._loss_lpips,
            'Ec': self._loss_ec,
            'GRAM': self._loss_gram,
            'P': self._loss_p_constraint,
            # 'ADV':self._loss_adv,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            # if loss_type == 'CLIP_COS':
            #     losses[loss_type] = tmp_loss.item()
            losses[loss_type] = tmp_loss
            loss = loss + float(weight) * tmp_loss
        return loss, losses
