from stylegan2.model import Generator, Discriminator, EqualLinear
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from build_loss import LossBuilder
from functools import partial
# from drive import open_url
from torch import optim
import clip
from torchvision import transforms, utils
import torchvision

class Activate(torch.nn.Module):
    def __init__(self):
        super(Activate, self).__init__()

        self.a = torch.nn.Parameter(torch.ones([1,512]))
        self.b = torch.nn.Parameter(torch.ones([1,512]))
        self.mean = torch.nn.Parameter(torch.zeros([1,512])*(-0.2))

    def forward(self,latent):

        map = torch.gt(latent, self.mean)
        latent = torch.where(map == 1, torch.exp(self.a * (latent - self.mean)) - 1,
                                      (-torch.exp(-1* self.b * (latent - self.mean)) + 1))
        return latent

class MyOptim(torch.nn.Module):
    def __init__(self, cache_dir, verbose=True):
        super(MyOptim, self).__init__()

        self.Generator = Generator(1024, 512, 8).cuda()
        self.verbose = verbose

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose: print("Loading Generator Network")
        self.Generator.load_state_dict(torch.load('stylegan2-ffhq-config-f.pt')['g_ema'])

        for param in self.Generator.parameters():
            param.requires_grad = False
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        self.gaussian_fit = torch.load("gaussian_fit_stylegan2_ffhq_clip_1.pt") # 是这个噢

        # else:
        #     if self.verbose: print("\tLoading Mapping Network")
        #     mapping = G_mapping().cuda()
        #
        #     # with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir=cache_dir, verbose=verbose) as f:
        #     mapping.load_state_dict(torch.load('mapping.pt'))
        #
        #     if self.verbose: print("\tRunning Mapping Network")
        #     with torch.no_grad():
        #         torch.manual_seed(0)
        #         latent = torch.randn((1000000,512),dtype=torch.float32, device="cuda")
        #         latent_out = torch.nn.LeakyReLU(5)(mapping(latent))
        #         self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
        #         torch.save(self.gaussian_fit,"gaussian_fit.pt") # stylegan原生 w
        #                                                         # stylegan生成图像的clip embedding
        #         if self.verbose: print("\tSaved \"gaussian_fit.pt\"")
        #         # mapping后加leakyrelu(5)然后再求出均值和方差
        # self.mapping = G_mapping().cuda()
        # self.mapping.load_state_dict(torch.load('mapping.pt'))

        # mapping = G_mapping().cuda()
        # mapping.load_state_dict(torch.load('mapping.pt'))
        # with torch.no_grad():
        #     torch.manual_seed(0)
        #     sample_latent = torch.randn((10000, 512), dtype=torch.float32, device="cuda")
        #     sample_latent = self.synthesis.get_latent(sample_latent)
        #     self.latent_avg = sample_latent.mean(0)  # , "std": latent_out.std(0)}
        #     del sample_latent
        with torch.no_grad():
            self.latent_avg = self.Generator.mean_latent(4096)
        self.mse = torch.nn.MSELoss()

    def forward(self, ref_im,
                ref_img_name,
                #wandb,
                seed,
                loss_str,
                loss_str2,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,  # default false
                bad_noise_layers,
                opt_name,
                learning_rate,
                steps,
                fc_every,
                lr_schedule,
                save_intermediate,
                **kwargs):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

        batch_size = ref_im.shape[0]

        loss_builder = LossBuilder(ref_im, loss_str, eps).cuda()
        loss_builder2 = LossBuilder(ref_im, loss_str2, eps).cuda()

        # Generate latent tensor
        if (tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            latent = torch.randn(
                (batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')

        var_list = [latent]  # + noise_vars

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9 * (1 - np.abs(x / steps - 1 / 2) * 2) + 1) / 10,
            'linear1cycledrop': lambda x: (9 * (
                    1 - np.abs(x / (0.9 * steps) - 1 / 2) * 2) + 1) / 10 if x < 0.9 * steps else 1 / 10 + (
                    x - 0.9 * steps) / (0.1 * steps) * (1 / 1000 - 1 / 10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        final_linear = torch.nn.Sequential(EqualLinear(512, 512,lr_mul=1),
                                           Activate(),
                                           ).cuda()
        decouple_mlp = torch.nn.Sequential(EqualLinear(512, 1024,lr_mul=1),
                                           torch.nn.Sigmoid(),
                                           EqualLinear(1024, 1024, lr_mul=1),
                                           torch.nn.Sigmoid(),
                                           ).cuda()
        couple_mlp = torch.nn.Sequential(EqualLinear(1024,1024,lr_mul=1),
                                         torch.nn.Sigmoid(),
                                         EqualLinear(1024, 512, lr_mul=1),
                                         torch.nn.Sigmoid()
                                         ).cuda()

        p_lambda = torch.nn.Parameter(torch.tensor(1.0))


        optim_mlp = optim.Adam(
            [{'params': p_lambda, 'lr': 0.1},
             {'params': final_linear.parameters()},
             {'params': decouple_mlp.parameters()},
             {'params': couple_mlp.parameters()}],
            # self.mapping.parameters(),
            lr=0.002,
            betas=(0, 0.99),
        )

        min_loss = np.inf
        min_lpips = np.inf
        best_summary = ""
        gen_im = None
        start_t = time.time()
        if self.verbose: print("Optimizing")
        optim_total_time = 0

        for j in range(steps):
            optim_start_t = time.time()
            opt.opt.zero_grad()

            if (tile_latent):
                clip_latent_in = latent
            else:
                clip_latent_in = latent

            # latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])
            clip_latent_in = clip_latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"]
            # clip_latent_in = self.lrelu(clip_latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            ref_clip=loss_builder.ref_clip_latent.unsqueeze(1)
            gen_clip=clip_latent_in
            decoupled_ref_clip = decouple_mlp((ref_clip+1)/2)
            decoupled_gen_clip = decouple_mlp(gen_clip)
            coupled_ref_clip = couple_mlp(decoupled_ref_clip)
            coupled_gen_clip = couple_mlp(decoupled_gen_clip)

            loss_decouple = ((1 - loss_builder.cos2(decoupled_ref_clip[:,:,:512], loss_builder.text_features_ref.unsqueeze(0)))[0] +
                            (1 - loss_builder.cos2(decoupled_gen_clip[:,:,:512], loss_builder.text_features_tar.unsqueeze(0)))[0]).mean()

            loss_couple = loss_builder.l1_loss(ref_clip,coupled_ref_clip) + loss_builder.l1_loss(gen_clip,coupled_gen_clip)
            loss_style = (1 - loss_builder.cos2(decoupled_ref_clip[:,:,512:], decoupled_gen_clip[:,:,512:]))[0][0] \
                        + self.mse(decoupled_ref_clip[:, :, 512:], decoupled_gen_clip[:, :, 512:]) * 100

            p_latent_in = final_linear(clip_latent_in)
            w_latent_in = self.lrelu(p_latent_in)

            gen_im, _ = self.Generator(w_latent_in, input_is_latent=True, noise=self.Generator.make_noise())
            gen_im = (gen_im + 1) / 2 # [-1,1] -> [0,1]

            # if j == 0:
            #     with torch.no_grad():
            #         out_path = Path(kwargs["output_dir"])
            #         sample = gen_im
            #         torchvision.utils.save_image(
            #             sample,
            #             f"%s/init.png" % (out_path),
            #             # nrow=1,
            #             # nrow=int(args.n_sample ** 0.5),
            #             # normalize=True,
            #             range=(-1, 1),
            #         )
            # Calculate Losses
            loss, loss_dict = loss_builder(clip_latent=clip_latent_in, latent=w_latent_in, gen_im=gen_im, searched_latent=latent.detach(),p_latent = p_latent_in,p_lambda = p_lambda)

            loss_dict['DECOUPLE'] = loss_decouple
            loss_dict['COUPLE'] = loss_couple
            loss_dict['STYLE'] = loss_style

            gen_clip_latent = loss_builder.clip_model.encode_image(loss_builder.preprocess(loss_builder.D_4(gen_im))).unsqueeze(1)
            gen_p_latent = final_linear(gen_clip_latent)
            gen_w_latent = self.lrelu(gen_p_latent)
            gen_im_2, _ = self.Generator(gen_w_latent.float(), input_is_latent=True,noise=self.Generator.make_noise())
            gen_im_2 = (gen_im_2 + 1) / 2

            loss_cycle = loss_builder._loss_l2(loss_builder.D_4(gen_im), loss_builder.D_4(gen_im_2))


            loss_dict['CYCLE'] = loss_cycle
            # loss_dict['CYCLE_CLIP'] = loss_cycle_clip
            loss_dict['TOTAL'] = loss + loss_cycle# + loss_cycle_clip

            loss_aaa = f'BEST ({j + 1}) | ' + ' | '.join([f'{x}: {y:.4f}' for x, y in loss_dict.items()])
            print(loss_aaa)
            if (loss < min_loss):
                min_loss = loss
                best_summary = f'BEST ({j + 1}) | ' + ' | '.join([f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.clone()

            if save_intermediate:
                yield (loss_builder.D_2(gen_im).clone().cpu().detach().clamp(0, 1))

            # (loss + loss_cycle + loss_decouple + loss_couple + 100*loss_style).backward(retain_graph=True)
            (loss + loss_cycle + loss_decouple + loss_couple + loss_style).backward(retain_graph=True)
            opt.step()
            scheduler.step()

            optim_iter_time = time.time() - optim_start_t
            optim_total_time = optim_total_time + optim_iter_time
            if j % fc_every == 0:
                optim_mlp.zero_grad()
                clip_latent_in = latent * self.gaussian_fit["std"] + self.gaussian_fit["mean"]
                ref_clip = loss_builder2.ref_clip_latent.unsqueeze(1)
                gen_clip = clip_latent_in
                decoupled_ref_clip = decouple_mlp((ref_clip+1)/2)
                decoupled_gen_clip = decouple_mlp(gen_clip)
                coupled_ref_clip = couple_mlp(decoupled_ref_clip)
                coupled_gen_clip = couple_mlp(decoupled_gen_clip)
                loss_decouple = (
                        (1 - loss_builder.cos2( decoupled_ref_clip[:, :, :512],(loss_builder.text_features_ref.unsqueeze(0)+1)/2  ))[0] +
                        (1 - loss_builder.cos2( decoupled_gen_clip[:, :, :512],(loss_builder.text_features_tar.unsqueeze(0)+1)/2  ))[0]
                ).mean()

                loss_couple = loss_builder2.l1_loss(ref_clip, coupled_ref_clip) + loss_builder2.l1_loss(gen_clip,
                                                                                                      coupled_gen_clip)
                loss_style = (1 - loss_builder2.cos2(decoupled_ref_clip[:, :, 512:], decoupled_gen_clip[:, :, 512:]))[0][0] \
                            + self.mse(decoupled_ref_clip[:, :, 512:], decoupled_gen_clip[:, :, 512:]) * 100

                p_latent_in = final_linear(clip_latent_in)
                w_latent_in = self.lrelu(p_latent_in)

                gen_im, _ = self.Generator(w_latent_in, input_is_latent=True, noise=self.Generator.make_noise())
                gen_im = (gen_im + 1) / 2

                loss2, loss_dict2 = loss_builder2(clip_latent=clip_latent_in, latent=w_latent_in, gen_im=gen_im,searched_latent=latent.detach(),p_latent = p_latent_in,p_lambda = p_lambda)#latent_avg=self.latent_avg)

                loss_dict['DECOUPLE'] = loss_decouple
                loss_dict['COUPLE'] = loss_couple
                loss_dict['STYLE'] = loss_style
                gen_clip_latent = loss_builder2.clip_model.encode_image(loss_builder2.preprocess(loss_builder2.D_4(gen_im))).unsqueeze(1)
                gen_p_latent = final_linear(gen_clip_latent)
                gen_w_latent = self.lrelu(gen_p_latent)
                gen_im_2, _ = self.Generator(gen_w_latent.float(), input_is_latent=True,noise=self.Generator.make_noise())
                gen_im_2 = (gen_im_2 + 1) / 2
                loss_cycle2 = loss_builder2._loss_l2(loss_builder2.D_4(gen_im), loss_builder2.D_4(gen_im_2))
                loss_dict2['CYCLE'] = loss_cycle2
                # loss_dict2['CYCLE_CLIP'] = loss_cycle_clip2
                loss_dict2['TOTAL'] = loss2 + loss_cycle2 # + loss_cycle_clip2
                #####

                # for k, v in loss_dict.items():
                #     wandb.log(
                #         {'mlp ' + k: v},
                #         # step=j
                #     )
                # (loss2 + loss_cycle2 + loss_decouple + loss_couple + 100*loss_style).backward(retain_graph=True)
                (loss2 + loss_cycle2 + loss_decouple + loss_couple + loss_style).backward(retain_graph=True)

                optim_mlp.step()
                relu = torch.nn.ReLU(inplace=False)
                p_lambda.data = relu(p_lambda.data)

        total_t = time.time() - start_t

        current_info = f' | time: {total_t:.1f} | optim_time: {optim_total_time:.1f} , {optim_total_time/total_t} | learning_time: {(total_t-optim_total_time):.1f} , {(total_t-optim_total_time)/total_t} | it/s: {(j + 1) / total_t:.2f} | batchsize: {batch_size}'
        if self.verbose: print(best_summary + current_info)

        yield (loss_builder.D_2(gen_im).clone().cpu().detach().clamp(0, 1), )
