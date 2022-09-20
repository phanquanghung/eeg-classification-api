# Import libraries
from pyexpat import model
from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGNetv4

import numpy as np
from flask import Flask, request, jsonify

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, EarlyStopping

app = Flask(__name__)

# cuda = torch.cuda.is_available()
# print('gpu: ', cuda)
# device = 'cuda' if cuda else 'cpu'

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# rng = RandomState(seed)

# device = torch.device("cuda")

buffer = np.empty((0, 0))

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"{super().__repr__()}, max_norm={self.max_norm}"

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )  # [out,in,H,W]
        return super(Conv2dWithConstraint, self).forward(x)

class TemporalSpatialConv(nn.Sequential):
    def __init__(
        self,
        in_chans,
        in_depth=1,
        F=8,
        D=2,
        kernel_length=64,
        drop_prob=0.5,
        bn_mm=0.01,
        bn_track=True,
    ):
        modules = [
            nn.Conv2d(
                in_depth,
                F,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, kernel_length // 2),
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F, 
                momentum=bn_mm, 
                affine=bn_track, 
                track_running_stats=bn_track,
            ),
            Conv2dWithConstraint(
                F,
                F * D,
                max_norm=1.0,
                kernel_size=(in_chans, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=F,
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F * D, 
                momentum=bn_mm, 
                affine=bn_track, 
                track_running_stats=bn_track,
            ),
            nn.ELU(inplace=True),
        ]

        super().__init__(*modules)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"{super().__repr__()}, max_norm={self.max_norm}"

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )  # [out,in]
        return super(LinearWithConstraint, self).forward(x)

class EEGClassifier(nn.Module):
    def __init__(self, Fz, n_classes, max_norm=None):
        super().__init__()

        if max_norm is None:
            self.linear = nn.Linear(Fz, n_classes, bias=True)
        else:
            self.linear = LinearWithConstraint(Fz, n_classes, max_norm=max_norm, bias=True)

    def forward(self, x):
        x = self.linear(x)

        return x

class EEGNet(nn.Module):
    def __init__(
        self,
        in_chans,
        in_samples,
        in_depth=1,
        Fs=(8,),
        Ds=(2,),
        F2=16,
        kernel_lengths=(64,),
        conv2_kernel_length=16,
        pool_size=8,
        drop_prob=0.5,
        bn_mm=0.01,
        bn_track=True,
    ):
        super().__init__()

        # multi-branch temporal-spatial conv ~ conv_1
        F1 = 0
        self.inception_block = nn.ModuleList()
        for F, D, kernel_length in zip(Fs, Ds, kernel_lengths):
            self.inception_block.append(
                TemporalSpatialConv(in_chans, in_depth, F, D, kernel_length, drop_prob, bn_mm, bn_track)
            )
            F1 += F * D

        self.pool_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), ceil_mode=False),
            nn.Dropout(p=drop_prob),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                F1, 
                F1, 
                kernel_size=(1, conv2_kernel_length), 
                stride=(1, 1), 
                padding=(0, conv2_kernel_length // 2), 
                groups=F1, 
                bias=not bn_track,
            ),
            nn.Conv2d(
                F1, 
                F2, 
                kernel_size=(1, 1), 
                stride=(1, 1), 
                padding=(0, 0), 
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F2,
                momentum=bn_mm,
                affine=bn_track, 
                track_running_stats=bn_track,
            ),
            nn.ELU(inplace=True),
        )

        self.pool_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size), ceil_mode=False),
            nn.Dropout(p=drop_prob),
        )

    def forward(self, x):
        if x.ndim == 3:  # [B,C,T]
            x = x.unsqueeze(1)  # [B,1,C,T]

        x = torch.cat([conv(x) for conv in self.inception_block], dim=1)  # [B,F1,1,T]
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        
        x = torch.flatten(x, start_dim=1)  # [B,*]
        
        return x


class EEGNetDecoder(nn.Module):
    def __init__(
        self,
        in_chans,
        in_samples,
        in_depth=1,
        Fz=16,
        Fs=(8,),
        Ds=(2,),
        F2=16,
        kernel_lengths=(64,),
        conv2_kernel_length=16,
        pool_size=8,
        drop_prob=0.5,
        bn_mm=0.01,
        bn_track=True,
    ):
        super().__init__()

        self.conv_trans_3 = nn.Sequential(
            nn.Conv2d(
                Fz, 
                F2, 
                kernel_size=(1, 1), 
                stride=(1, 1), 
                padding=(0, 0), 
                bias=not bn_track,
            ),
            nn.ConvTranspose2d(
                F2,
                F2,
                kernel_size=(1, in_samples//(4 * pool_size)),
                stride=(1, 1),
                padding=(0, 0),
                groups=F2,
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F2,
                momentum=bn_mm,
                affine=bn_track, 
                track_running_stats=bn_track,
            ),
            nn.ELU(inplace=True),
            nn.Dropout(p=drop_prob),
        )

        F1 = sum(F * D for F, D in zip(Fs, Ds))
        self.conv_trans_2 = nn.Sequential(
            nn.Conv2d(
                F2,
                F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=not bn_track,
            ),
            nn.ConvTranspose2d(  # out == in * stride(pool_size)
                F1,
                F1,
                kernel_size=(1, conv2_kernel_length),
                stride=(1, pool_size),
                padding=(0, (conv2_kernel_length - pool_size) // 2),
                groups=F1,
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F1,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.ELU(inplace=True),
            nn.Dropout(p=drop_prob),
        )

        # multi-branch temporal-spatial conv transpose
        self.inception_trans_block = nn.ModuleList()
        for F, D, kernel_length in zip(Fs, Ds, kernel_lengths):
            self.inception_trans_block.append(
                TemporalSpatialConvTrans(in_chans, in_depth, F, D, kernel_length, drop_prob, bn_mm, bn_track)
            )

        self.Fi = [0]
        for F, D in zip(Fs, Ds):
            self.Fi.append(self.Fi[-1] + F * D)

        self.in_chans = in_chans
        self.in_depth = in_depth

    def forward(self, x):  # [B,Fz]
        x = x.unsqueeze(2).unsqueeze(3)  # [B,Fz,1,1]
        x = self.conv_trans_3(x)
        x = self.conv_trans_2(x)  # [B,F1,1,T//4]        
        x = torch.sum(torch.stack([conv_trans(x[:, self.Fi[i] : self.Fi[i + 1]]) for i, conv_trans in enumerate(self.inception_trans_block)]), dim=0)  # [B,in_chans*in_depth,1,T]
        x = x.reshape(x.size(0), self.in_depth, self.in_chans, -1).contiguous()

        return x

def init_weights(module: nn.Module, use_xavier: bool = True, use_uniform: bool = True, gain: float = 1.0, nonlinearity: str = "leaky_relu"):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            if use_xavier:
                if use_uniform:
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                if use_uniform:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)

            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.0)
        
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, val=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.0)

class MotorImageryModule(pl.LightningModule):
    def __init__(self, n_classes, n_channels, n_samples, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples

        drop_prob = self.hparams.get("drop_prob", 0.0)

        self.encoder = EEGNet(
            in_chans=n_channels,
            in_samples=n_samples,
            Fs=(8,),
            Ds=(4,),
            F2=32,
            kernel_lengths=(64,),
            conv2_kernel_length=16,
            pool_size=8,
            drop_prob=drop_prob,
            bn_mm=0.01,
            bn_track=True,
        )  # [B,F2*(n_samples//32)]
        # self.encoder = EEG_ITNet(
        #     in_chans=n_channels,
        #     in_samples=n_samples,
        #     Fs=(8,4,2,),
        #     Ds=(2,2,2,),
        #     F2=28,
        #     kernel_lengths=(64,32,16,),
        #     tc_kernel_length=5,
        #     n_tc_layers=4,
        #     pool_size=8,
        #     drop_prob=drop_prob,
        #     bn_mm=0.01,
        #     bn_track=True,
        # )

        with torch.no_grad():
            x = torch.zeros((1, 1, n_channels, n_samples), dtype=torch.float32)
            x = self.encoder(x)
            flatten_size = x.size(1)
        
        self.clf = EEGClassifier(flatten_size, n_classes, max_norm=0.25)  # [B,n_classes]

        # Fz = 16
        # self.decoder = EEGNetDecoder(
        #     in_chans=n_channels,
        #     in_samples=n_samples,
        #     Fz=Fz,
        #     kernel_lengths=(64,32,16,),
        #     conv2_kernel_length=16,
        #     pool_size=8,
        #     drop_prob=drop_prob,
        #     bn_mm=0.01,
        #     bn_track=True,
        # )

        # # VAE
        # self.vae_mean = nn.Linear(flatten_size, Fz, bias=True)
        # self.vae_logvar = nn.Linear(flatten_size, Fz, bias=True)

        # load pretrained model
        pretrained_checkpoint = self.hparams.get("pretrained_checkpoint", None)
        if pretrained_checkpoint is not None:
            print(f"loading pretrained checkpoint {pretrained_checkpoint}...")

            state_dict = torch.load(pretrained_checkpoint, map_location=self.device)["state_dict"]

            if self.hparams.get("reinit_clf", False):
                state_dict.pop("clf.linear.weight")
                state_dict.pop("clf.linear.bias")
                self.load_state_dict(state_dict, strict=False)  # ignore clf

                init_weights(self.clf, use_xavier=True, use_uniform=True, gain=1.0, nonlinearity="leaky_relu")

            else:
                self.load_state_dict(state_dict, strict=True)

            if self.hparams.get("train_clf_only", False):
                for p in self.encoder.parameters():
                    p.requires_grad = False

                for m in self.encoder.modules():  # disable dropout
                    if isinstance(m, nn.Dropout):
                        m.p = 0

        else:
            print(f"initializing module's weights...")
            init_weights(self, use_xavier=True, use_uniform=True, gain=1.0, nonlinearity="leaky_relu")

        # loss' params
        self.src_class_weight = self.hparams.get("src_class_weight", None)
        self.tgt_class_weight = self.hparams.get("tgt_class_weight", None)

        self.src_loss_scale = self.hparams.get("src_loss_scale", 1.0)
        self.tgt_loss_scale = self.hparams.get("tgt_loss_scale", 1.0)

        self.smooth_label = self.hparams.get("smooth_label", 0.0)

        self.use_focal_loss = self.hparams.get("use_focal_loss", False)
        self.focal_loss_gamma = self.hparams.get("focal_loss_gamma", 2.0)
        
        # self.vae_loss_scale = self.hparams.get("vae_loss_scale", 1.0)
        # self.kld_betas = frange_cycle_linear(start=0.0, stop=1.0, 
        #                                      n_epoch=self.hparams.get("epochs", 1), 
        #                                      n_cycle=self.hparams.get("kld_betas_n_cycles", 10), 
        #                                      ratio=self.hparams.get("kld_betas_ratio", 1.0))

        # mixup
        self.mixup_alpha = self.hparams.get("mixup_alpha", 0.0)
        self.mixup_cross_label = self.hparams.get("mixup_cross_label", True)
        self.rng = np.random.RandomState(seed=self.hparams.get("rng_seed", 42))

    def configure_optimizers(self):
        lr = self.hparams.get("lr", 1e-3)
        epochs = self.hparams.get("epochs", 1)

        # optimizer = optim.AdamW(
        optimizer = EAdam(
            self.parameters(),
            lr=lr,
            weight_decay=self.hparams.get("weight_decay", 0.0),
            amsgrad=False,
        )

        if self.hparams.get("use_lr_sch", False):
            optimizer_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "name": "optim/lr",
                    "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.hparams.get("min_lr", 0)),
                    # "scheduler": optim.lr_scheduler.OneCycleLR(optimizer, 
                    #                                            max_lr=lr, 
                    #                                            epochs=epochs, 
                    #                                            steps_per_epoch=self.hparams.get("steps_per_epoch", 1), 
                    #                                            pct_start=self.hparams.get("pct_start", 0.1)),
                    # "interval": "step",
                },
            }

        else:
            optimizer_dict = {
                "optimizer": optimizer,
            }

        return optimizer_dict

    def compute_loss(self, x, y, class_weight=None, reduce=True):
        x = F.log_softmax(x, dim=1)

        if self.use_focal_loss:
            return focal_loss(x, y, weight=class_weight, smooth=self.smooth_label, gamma=self.focal_loss_gamma, reduce=reduce)
        else:
            return nll_loss(x, y, weight=class_weight, smooth=self.smooth_label, reduce=reduce)

    def forward(self, x):
        e = self.encoder(x)
        logit = self.clf(e)

        return logit, e
        
        # # ----------VAE----------
        # z_mean = self.vae_mean(e)
        # z_logvar = self.vae_logvar(e)

        # std = torch.exp(0.5 * z_logvar)
        # z = z_mean + torch.randn_like(std) * std

        # return logit, z, z_mean, z_logvar

    def predict(self, x):
        return torch.argmax(F.log_softmax(self(x)[0], dim=1), dim=1)

    def epoch_end(self, outputs, mode="train"):
        y_true = torch.cat([output["y_true"] for output in outputs], dim=0).detach().cpu().numpy()
        y_pred = torch.cat([output["y_pred"] for output in outputs], dim=0).detach().cpu().numpy()

        # # plot generated sample
        # x = outputs[0]["x"].detach().cpu().numpy()  # [T,]
        # r = outputs[0]["r"].detach().cpu().numpy()  # [T,]
        # _ = self.logger.experiment.add_figure(f"{mode}/eeg_reconstruction", reconstruction_figure(x=x, r=r))

        # plot confusion matrix
        _ = self.logger.experiment.add_figure(f"{mode}/confusion_matrix", confusion_matrix_figure(y_true=y_true, y_pred=y_pred, n_classes=self.n_classes))

        # plot metrics
        scores = compute_scores(y_true=y_true, y_pred=y_pred)
        if mode != "train":
            scores["loss"] = float(torch.stack([output["loss"] for output in outputs]).mean())

        for k, v in scores.items():
            self.log(f"{mode}/{k}", v)

    def shared_step(self, x, y, class_weight=None):
        bsz = x.size(0)

        if self.mixup_alpha > 0:
            if self.mixup_cross_label:
                # mixup all classes trials
                lam = torch.tensor(self.rng.beta(self.mixup_alpha, self.mixup_alpha, bsz)).float().to(self.device)
                lam = lam.reshape(-1, 1, 1)
                _lam = 1.0 - lam

                perm = torch.tensor(self.rng.permutation(bsz)).long()
                y_perm = y[perm]

                mixed_x = lam * x + _lam * x[perm]

                logit = self(mixed_x)[0]

                loss = lam * self.compute_loss(logit, y, class_weight, reduce=False) + _lam * self.compute_loss(logit, y_perm, class_weight, reduce=False)
                loss = loss.mean()

            else:
                # mixup single class trials
                lam = torch.tensor(self.rng.beta(self.mixup_alpha, self.mixup_alpha, x.size(0))).float().to(self.device)
                lam = lam.reshape(-1, 1, 1)
                _lam = 1.0 - lam

                perm = torch.zeros(bsz).long()
                for c in torch.unique(y):
                    idx = torch.where(y == c)[0].cpu()
                    perm[idx] = idx[torch.randperm(idx.size(0))]

                mixed_x = lam * x + _lam * x[perm]

                logit = self(mixed_x)[0]

                loss = self.compute_loss(logit, y, class_weight)

        else:
            logit = self(x)[0]

            loss = self.compute_loss(logit, y, class_weight)

            # # ----------VAE----------
            # logit, z, z_mean, z_logvar = self(x)  # z*: [B,Fz]

            # loss_clf = self.compute_loss(logit, y, class_weight)

            # r = self.decoder(z).squeeze(1)  # [B,C,T]
            # loss_rec = F.mse_loss(r, x, reduction="none").sum(dim=(1,2))
            # loss_kld = 0.5 * torch.sum(z_mean.pow(2) + z_logvar.exp() - z_logvar - 1, dim=1)

            # kld_beta = self.kld_betas[self.current_epoch] if self.training else 1
            # loss_vae = torch.mean(loss_rec + kld_beta * loss_kld)  # ???
            
            # log_tag = 'train' if self.training else 'valid'
            # self.log(f"{log_tag}/loss_clf", loss_clf)
            # self.log(f"{log_tag}/loss_rec", loss_rec.mean())
            # self.log(f"{log_tag}/loss_kld", loss_kld.mean())

            # loss = loss_clf + self.vae_loss_scale * loss_vae

        return loss

    def training_step(self, batch, batch_idx, **kwargs):
        # -----------single source-------------
        x, y = batch  # X*: [B,C,T], Y*: [B,]

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.predict(x)

        # # ---------VAE----------
        # logit, z, _, _ = self(x)
        # y_pred = torch.argmax(F.log_softmax(logit, dim=1), dim=1)
        # r = self.decoder(z).squeeze(1)  # [B,C,T]

        loss = self.shared_step(x, y, self.src_class_weight)
        self.log("train/loss", loss)

        # rand_chs = self.rng.randint(0, n_channels, (2,))  # select random channels
        return {
            "loss": loss,
            "y_true": y,
            "y_pred": y_pred,
            # "x": x[0, rand_chs],  # [T,2]
            # "r": r[0, rand_chs],  # [T,2]
        }

        # # -----------multiple sources----------
        # (xsrc, ysrc), (xtgt_labeled, ytgt_labeled) = batch

        # xsrc = xsrc.to(self.device)
        # ysrc = ysrc.to(self.device)
        # xtgt_labeled = xtgt_labeled.to(self.device)
        # ytgt_labeled = ytgt_labeled.to(self.device)

        # ytgt_pred = self.predict(xtgt_labeled)

        # loss = self.src_loss_scale * self.shared_step(xsrc, ysrc, self.src_class_weight) + self.tgt_loss_scale * self.shared_step(xtgt_labeled, ytgt_labeled, self.tgt_class_weight)
        # self.log("train/loss", loss)

        # return {
        #     "loss": loss,
        #     "y_true": ytgt_labeled,
        #     "y_pred": ytgt_pred,
        # }

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, mode="train")

    def validation_step(self, batch, batch_idx, **kwargs):
        x, y = batch  # X*: [B,C,T], Y*: [B,]

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.predict(x)

        # # ---------VAE----------
        # logit, z, _, _ = self(x)
        # y_pred = torch.argmax(F.log_softmax(logit, dim=1), dim=1)
        # r = self.decoder(z).squeeze(1)  # [B,C,T]

        loss = self.shared_step(x, y)

        # rand_chs = self.rng.randint(0, n_channels, (2,))  # select random channels
        return {
            "loss": loss,
            "y_true": y,
            "y_pred": y_pred,
            # "x": x[0, rand_chs],  # [T,]
            # "r": r[0, rand_chs],  # [T,]
        }

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, mode="valid")

    def test_step(self, batch, batch_idx, **kwargs):
        return self.validation_step(batch, batch_idx, **kwargs)

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, mode="test")

# Load the model
model = MotorImageryModule(
        n_classes=4,
        n_channels=28,
        n_samples=256,
    )
model.load_state_dict(torch.load('EEGNet8,4_nonEA.ckpt', map_location=torch.device('cpu'))["state_dict"])
# model.to(device)
model.eval()

def update_buffer(data_np):
    global buffer
    if buffer.shape[0] == 0:
        buffer = data_np
    else:
        buffer = np.append(buffer, data_np, axis = 0)

def preprocess(full_buffer):
    # add batch dims
    full_buffer = np.expand_dims(full_buffer, axis=0)
    # full_buffer = np.expand_dims(full_buffer, axis=0)
    data_tensor = torch.from_numpy(full_buffer).float()
	# convert type to torch tensor
	# # input = torch.from_numpy(input)
	# reshape input size
    data_tensor = data_tensor.permute(0, 2, 1)
    return data_tensor

def server_predict(input):
    output = model.predict(input)
    # output = F.softmax(output, dim=1)
    return output

@app.route('/api',methods=['POST'])
def update_status():
    global buffer
    # Get the data from the POST request.
    data = request.get_json(force=True)
    data_np = np.array(data['eeg'])

    # Preprocessing
    # ['FT9', 'PO9', 'PO10', 'FT10']
    data_np = np.delete(data_np, [8, 14, 19, 25], axis=1)
    # data_np = data_np.astype('float32') * 1e3

    update_buffer(data_np)
    if buffer.shape[0] == 256:
        input = preprocess(buffer)
        buffer = np.empty((0, 0))
        output = server_predict(input)
        # return jsonify({'output': output.tolist()})
        if output == 0:
            return 'leg'
        elif output == 1:
            return 'right hand'
        elif output == 2:
            return 'left hand'
        elif output == 3:
            return 'rest'
    else:
        return 'collecting data'
    # input_cat = torch.cat((input_cat, input_tensor), 0)


if __name__ == '__main__':
    # load_model()
    # try:
    app.run(port=5001, debug=True)
    # except:
    #     print("Server is exited unexpectedly. Please contact server admin.")