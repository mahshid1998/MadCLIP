import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class MCNNFC(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(MCNNFC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=1, stride=1),
        )
        self.relu = nn.LeakyReLU(inplace=False)
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.norm1 = nn.InstanceNorm1d(c_in)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.cnn(x)  # Apply 1D convolution
        # Permute back to [290, batch_size, 768] for LayerNorm
        x = x.permute(2, 0, 1)  # Shape: [seq_len, batch_size, bottleneck]
        x = self.norm1(x)
        x = self.relu(x)
        y = self.fc1(x)
        z = self.fc2(x)
        return y, z


class MFCFC(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(MFCFC, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.norm1 = nn.InstanceNorm1d(c_in)
        self.fc2 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        y = self.fc2(x)
        z = self.fc3(x)
        return y, z


class MViTFC(nn.Module):
    def __init__(self, c_in, bottleneck, num_heads=8, dropout=0.1):
        super(MViTFC, self).__init__()

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=c_in,
            nhead=num_heads,
            dim_feedforward=c_in,
            dropout=dropout,
            activation='relu'  # Use LeakyReLU as the activation function
        )
        # Bottleneck fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Input: [290, batch_size, 1024]
        x = x.permute(1, 0, 2)  # Permute to [batch_size, 290, 1024] for Transformer
        x = self.transformer_encoder(x)  # Apply Vision Transformer encoder
        x = x.permute(1, 0, 2)  # Back to [290, batch_size, 1024]
        y = self.fc(x)  # Apply bottleneck fully connected layers
        z = self.fc1(x)
        return y, z

class CLIP_Inplanted(nn.Module):
    def __init__(self,args, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = args.features_list
        # self.res_mood = args.adapter_res_mood
        self.img_size = args.img_size
        ###################
        # adapters set up #
        ###################
        if args.visionA == "MCNNFC":
            self.normal_det_adapters = nn.ModuleList([MCNNFC(1024, bottleneck=768) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList( [MCNNFC(1024, bottleneck=768) for i in range(len(self.features))] )
            print("MCNNFC")
        elif args.visionA == "MFCFC":
            self.normal_det_adapters = nn.ModuleList([MFCFC(1024, bottleneck=768) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList( [MFCFC(1024, bottleneck=768) for i in range(len(self.features))] )
            print("MFCFC")
        elif args.visionA == "MViTFC":
            self.normal_det_adapters = nn.ModuleList([MViTFC(1024, bottleneck=768) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList( [MViTFC(1024, bottleneck=768) for i in range(len(self.features))] )
            print("MViTFC")
        self.all_adapters_optimizer = torch.optim.Adam(
            [
                {'params': self.normal_det_adapters.parameters(), 'lr': args.learning_rate},
                {'params': self.abnormal_det_adapters.parameters(), 'lr': args.learning_rate},
            ],
            betas=(0.5, 0.999)
        )

        ###################
        # contrast set up #
        ###################
        self.contrast_mood = args.contrast_mood
        if self.contrast_mood == "no":
            self.contrast = lambda a, b: (a)
        elif self.contrast_mood== "yes":
            self.contrast = lambda a, b: (a - b)
        else:
            print("ERROR, no such a contrast mood")


    def forward(self, x, text_features):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        det_scores = []
        seg_scores = []
        for i in range(24):
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:
                normal_f_det_i, normal_f_seg_i = self.normal_det_adapters[self.features.index(i+1)](x)
                abnormal_f_det_i, abnormal_f_seg_i = self.abnormal_det_adapters[self.features.index(i+1)](x)

                # reshape from [290,1,768] to [1,290,768]
                normal_f_det_i = normal_f_det_i.permute(1, 0, 2)
                normal_f_seg_i = normal_f_seg_i.permute(1, 0, 2)
                abnormal_f_det_i = abnormal_f_det_i.permute(1, 0, 2)
                abnormal_f_seg_i = abnormal_f_seg_i.permute(1, 0, 2)

                # remove extra dimension [1,290,768] to [289,768]
                normal_f_det_i = normal_f_det_i[:,1:,:]
                normal_f_seg_i = normal_f_seg_i[:,1:,:]
                abnormal_f_det_i = abnormal_f_det_i[:,1:,:]
                abnormal_f_seg_i = abnormal_f_seg_i[:,1:,:]

                # normalizing adapted features
                normal_f_det_i = normal_f_det_i /normal_f_det_i.norm(dim=-1, keepdim=True)
                normal_f_seg_i = normal_f_seg_i /normal_f_seg_i.norm(dim=-1, keepdim=True)
                abnormal_f_det_i = abnormal_f_det_i /abnormal_f_det_i.norm(dim=-1, keepdim=True)
                abnormal_f_seg_i = abnormal_f_seg_i /abnormal_f_seg_i.norm(dim=-1, keepdim=True)

                #####################################
                # Dual branch on detection features #
                #####################################
                #  text features = [768,2]: [:,0] -> t_n , [:,1] -> t_ab
                #   ---> output: S_n,i = (O_n,i * t_n) - (O_n.i * t_ab)
                sim_det_normal = self.dual_contrast(normal_f_det_i, text_features[:, 0], text_features[:, 1])

                # ---> output: S_ab,i = (O_ab,i * t_ab) - (O_ab.i * t_n)
                sim_det_abnormal = self.dual_contrast(abnormal_f_det_i, text_features[:, 1], text_features[:, 0])

                # ---> output: S_i = [S_n,i, S_ab,i]
                det_scores_cur = torch.cat([sim_det_normal, sim_det_abnormal], dim=-1)
                det_scores.append(det_scores_cur)

                ##########################################
                #  Dual branch on Segmentation features  #
                ##########################################
                # normality branch
                sim_seg_normal = self.dual_contrast(normal_f_seg_i, text_features[:,0], text_features[:,1])

                # abnormality branch
                sim_seg_abnormal = self.dual_contrast(abnormal_f_seg_i, text_features[:,1], text_features[:,0])
                seg_scores_cur = torch.cat([sim_seg_normal, sim_seg_abnormal], dim=-1)  # shape: [B, 2(channels), img_size, img_size]

                seg_scores.append(seg_scores_cur)



        x = x.permute(1, 0, 2)

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, det_scores, seg_scores


    def dual_contrast(self, features, same_text, opposite_text):
        same_view = (features @ same_text.unsqueeze(-1)) #[batch, 289,1]
        cross_view = ( features @ opposite_text.unsqueeze(-1)) #[batch, 289,1]

        return self.contrast(same_view, cross_view)