import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchvision.models import resnet50

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../cem'))
from models.cbm import CBM_SSL
import train.utils as utils
from utils import visualize_and_save_heatmaps
from cem.metrics.accs import compute_accuracy


class SSCBM(CBM_SSL):
    def __init__(
            self,
            n_concepts,
            n_tasks,
            emb_size=16,
            training_intervention_prob=0.25,
            embedding_activation="leakyrelu",
            shared_prob_gen=True,
            concept_loss_weight=1,
            concept_loss_weight_labeled=1,
            concept_loss_weight_unlabeled=5,
            task_loss_weight=1,

            c2y_model=None,
            c2y_layers=None,
            c_extractor_arch=utils.wrap_pretrained_model(resnet50),
            output_latent=False,

            optimizer="adam",
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            weight_loss=None,
            task_class_weights=None,
            tau=1,

            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_policy=None,
            output_interventions=False,
            use_concept_groups=False,

            top_k_accuracy=None,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(active_intervention_values)
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(inactive_intervention_values)
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        self.resnet_out_features = list(self.pre_concept_model.modules())[-1].out_features

        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            if embedding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[torch.nn.Linear(self.resnet_out_features, 2 * emb_size)])
                )
            elif embedding_activation == "sigmoid":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(self.resnet_out_features, 2 * emb_size),
                        torch.nn.Sigmoid(),
                    ])
                )
            elif embedding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(self.resnet_out_features, 2 * emb_size),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embedding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(self.resnet_out_features, 2 * emb_size),
                        torch.nn.ReLU(),
                    ])
                )
            if self.shared_prob_gen and len(self.concept_prob_generators) == 0:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))

        if c2y_model is None:
            units = [n_concepts * emb_size] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        self.sigmoid = torch.nn.Sigmoid()

        self.loss_concept_labeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_concept_unlabeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights)
        )
        self.concept_loss_weight_labeled = concept_loss_weight_labeled
        self.concept_loss_weight_unlabeled = concept_loss_weight_unlabeled
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.tau = tau
        self.use_concept_groups = use_concept_groups

        self.fc = nn.Linear(512, self.emb_size)
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def unlabeled_image_encoder(self, x):
        # self.pre_concept_model resnet34
        x = self.pre_concept_model.conv1(x)
        x = self.pre_concept_model.bn1(x)
        x = self.pre_concept_model.relu(x)
        x = self.pre_concept_model.maxpool(x)

        x = self.pre_concept_model.layer1(x)
        x = self.pre_concept_model.layer2(x)
        x = self.pre_concept_model.layer3(x)
        x = self.pre_concept_model.layer4(x)
        x = x.transpose(1, 3)
        x = self.fc(x)
        return x

    def _after_interventions(
            self,
            prob,
            pos_embeddings,
            neg_embeddings,
            intervention_idxs=None,
            c_true=None,
            train=False,
            competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
                (c_true is not None) and
                (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true, intervention_idxs

    def compute_concept_embedding(self, pos_embeddings, neg_embeddings, concept_probs):
        """
        统一的概念 embedding 组合函数。

        原始图像路径和合成特征路径都必须调用这个函数，避免两条路径各自实现不同逻辑。
        对每个概念，SSCBM 的定义是：

            e_k = e_k^+ * p_k + e_k^- * (1 - p_k)

        Args:
            pos_embeddings: [batch, n_concepts, emb_size]，正概念 embedding。
            neg_embeddings: [batch, n_concepts, emb_size]，负概念 embedding。
            concept_probs: [batch, n_concepts]，概念概率。

        Returns:
            c_embedding: [batch, n_concepts, emb_size]，组合后的概念 embedding。
        """
        return (
            pos_embeddings * torch.unsqueeze(concept_probs, dim=-1) +
            neg_embeddings * (1 - torch.unsqueeze(concept_probs, dim=-1))
        )

    def predict_task_from_concept_embedding(self, c_embedding):
        """
        统一的 concept-to-label 预测头。

        原始 _forward 和 predict_from_features 都通过同一个 c2y_model 得到任务 logits，
        从而保证真实图像和合成特征共享完全一致的概念到类别预测流程。
        """
        c_pred = c_embedding.view((-1, self.emb_size * self.n_concepts))
        return c_pred, self.c2y_model(c_pred)

    def predict_concepts_from_feature_embedding(self, image_feature, c_embedding):
        """
        根据空间特征图和概念 embedding 计算概念概率。

        Args:
            image_feature: [batch, H, W, emb_size]，SSCBM 空间特征图。
            c_embedding: [batch, n_concepts, emb_size]，组合后的概念 embedding。

        Returns:
            c_pred_unlabeled: [batch, n_concepts]，由 heatmap 池化得到的概念概率。
        """
        heatmap = []
        for i in range(len(image_feature)):
            heatmap.append(torch.matmul(image_feature[i], c_embedding[i].transpose(0, 1)))
        heatmap = torch.stack(heatmap).permute(0, 3, 1, 2)

        c_pred_unlabeled = self.pooling(heatmap).squeeze()
        if c_pred_unlabeled.dim() == 1:
            c_pred_unlabeled = c_pred_unlabeled.unsqueeze(0)
        return self.sigmoid(c_pred_unlabeled)

    def _forward(
            self,
            x,
            c=None,
            y=None,
            l=None,
            train=False,
            latent=None,
            intervention_idxs=None,
            competencies=None,
            prev_interventions=None,
            output_embeddings=False,
            output_latent=None,
            output_interventions=None,
            output_feature_map=False,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)  # [batch_size, 299, 299] -> [batch_size, resnet_out_features]
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)  # [batch_size, resnet_out_features] -> [batch_size, 2 * emb_size]
                prob = prob_gen(context)  # [batch_size, 2 * emb_size] -> [batch_size, 1]
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sigmoid(prob))
            c_sem = torch.cat(c_sem, dim=-1)  # [batch_size, 1, n_concepts] -> [batch_size, n_concepts]
            contexts = torch.cat(contexts, dim=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        if (intervention_idxs is None) and (c is not None) and (self.intervention_policy is not None):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=None,
            )
        else:
            c_int = c

        if not train:
            intervention_idxs = self._standardize_indices(intervention_idxs=intervention_idxs, batch_size=x.shape[0])

        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, :self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )

        # 使用统一函数组合正/负概念 embedding。
        # 这样真实图像路径和合成特征路径共享同一个 concept embedding 定义。
        c_embedding = self.compute_concept_embedding(
            contexts[:, :, :self.emb_size],
            contexts[:, :, self.emb_size:],
            probs,
        )  # [batch_size, n_concepts, D]
        c_pred, y = self.predict_task_from_concept_embedding(c_embedding)

        # image_feature = self.pre_concept_model(x)  # [batch_size, resnet_out_features]
        # c_pred_unlabeled = self.cross_attn(image_feature, c_embedding)

        image_feature = self.unlabeled_image_encoder(x)

        # 使用统一函数从空间特征图计算概念概率，避免 _forward 与 predict_from_features 分叉。
        c_pred_unlabeled = self.predict_concepts_from_feature_embedding(image_feature, c_embedding)

        tail_results = []
        if output_interventions:
            if intervention_idxs is not None and isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(intervention_idxs).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])
        if output_feature_map:
            tail_results.append(image_feature)

        return tuple([c_sem, c_pred, c_pred_unlabeled, y] + tail_results)

    def _run_step(
            self,
            batch,
            batch_idx,
            train=False,
            intervention_idxs=None,
    ):
        x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions = self._unpack_batch(batch)

        nbr_w_ = nbr_w.unsqueeze(-1).repeat(1, 1, nbr_c.size(2))
        c_pseudo = nbr_c * nbr_w_
        c_pseudo = torch.sum(c_pseudo, dim=1) / nbr_w.size(1)

        outputs = self._forward(
            x,
            c=c,
            y=y,
            l=l,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
        )
        c_sem, c_pred_labeled, c_pred_unlabeled, y_pred = outputs[0], outputs[1], outputs[2], outputs[3]

        task_loss = self.loss_task(y_pred, y)
        task_loss_scalar = task_loss.detach()

        concept_loss_labeled = self.loss_concept_labeled(c_sem[l], c[l])
        concept_loss_scalar_labeled = concept_loss_labeled.detach()

        c_pred_unlabeled = c_pred_unlabeled.float()
        c_pseudo = c_pseudo.float()
        
        if (~l).any():
            concept_loss_unlabeled = self.loss_concept_unlabeled(c_pred_unlabeled[~l], c_pseudo[~l])
            concept_loss_scalar_unlabeled = concept_loss_unlabeled.detach()
        else:
            concept_loss_unlabeled = torch.tensor(0.0, device=x.device, requires_grad=True)
            concept_loss_scalar_unlabeled = torch.tensor(0.0, device=x.device)

        loss = (task_loss
                + self.concept_loss_weight_labeled * concept_loss_labeled
                + self.concept_loss_weight_unlabeled * concept_loss_unlabeled)

        # compute accuracy
        (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1) = compute_accuracy(c_sem, y_pred, c, y)
        result = {
            "c_acc": c_acc,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_acc": y_acc,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "c_loss_labeled": concept_loss_scalar_labeled,
            "c_loss_unlabeled": concept_loss_scalar_unlabeled,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_acc + y_acc) / 2,
        }
        return loss, result

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            print(f"================================Epoch {self.current_epoch}===============================")
        loss, result = self._run_step(batch, batch_idx, train=True)
        for name, val in result.items():
            if name in ['c_f1', 'y_auc', 'avg_c_y_acc', 'y_f1']:
                continue
            self.log(name, val, prog_bar=True)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result['c_acc'],
                "c_auc": result['c_auc'],
                "c_f1": result['c_f1'],
                "y_accuracy": result['y_acc'],
                "y_auc": result['y_auc'],
                "y_f1": result['y_f1'],
                "concept_loss_labeled": result['c_loss_labeled'],
                "concept_loss_unlabeled": result['c_loss_unlabeled'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
            },
        }

    def predict_step(
            self,
            batch,
            batch_idx,
            intervention_idxs=None,
            dataloader_idx=0,
    ):
        x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions = self._unpack_batch(batch)
        return self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            l=l,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_interventions=True
        )

    def predict_from_features(self, image_feature, pos_embeddings, neg_embeddings, concept_probs, task_concept_probs=None):
        """
        根据合成空间特征和概念上下文，复用 SSCBM 原始预测头。

        为了解决“合成特征路径与原始 _forward 逻辑不一致”的问题，这里不再使用
        c_embedding * c_pred_unlabeled 这类近似逻辑，而是严格复用原始 SSCBM 定义：

            c_embedding = pos * prob + neg * (1 - prob)
            y = c2y(flatten(c_embedding))

        因此真实图像路径和合成特征路径共享同一个 concept-to-label prediction pipeline。

        重要说明：
            concept_probs 用于先构造一个基础 concept embedding，再从 image_feature
            预测合成特征自己的概念概率 c_pred_unlabeled。

            旧实现直接用 concept_probs 计算任务 logits。这样任务预测几乎不看
            image_feature，step2 中传入基座 c_sem_b 时，合成样本的目标类概率会严重偏向
            基座语义。现在允许传入 task_concept_probs；如果不传，则仍保持旧行为。
            D-CGFS 新合成路径会把合成特征预测出的概念概率和基座概率混合后传入
            task_concept_probs，使任务 logits 真正受到合成特征语义影响。

        Args:
            image_feature: Tensor of shape [batch_size, H, W, D]，合成后的空间特征图。
            pos_embeddings: Tensor of shape [batch_size, n_concepts, D]，正概念 embedding。
            neg_embeddings: Tensor of shape [batch_size, n_concepts, D]，负概念 embedding。
            concept_probs: Tensor of shape [batch_size, n_concepts]，用于组合正负 embedding 的概念概率。
            task_concept_probs: 可选，用于任务分类头的概念概率。

        Returns:
            c_pred_unlabeled: 概念层预测概率, shape [batch_size, n_concepts]
            y: 最终任务类别预测 logits, shape [batch_size, n_tasks]
        """
        c_embedding = self.compute_concept_embedding(
            pos_embeddings,
            neg_embeddings,
            concept_probs,
        )
        c_pred_unlabeled = self.predict_concepts_from_feature_embedding(image_feature, c_embedding)
        if task_concept_probs is None:
            task_embedding = c_embedding
        else:
            task_embedding = self.compute_concept_embedding(
                pos_embeddings,
                neg_embeddings,
                task_concept_probs,
            )
        _, y = self.predict_task_from_concept_embedding(task_embedding)
        return c_pred_unlabeled, y

    def plot_heatmap(
            self,
            x,
            x_show=None,
            c=None,
            y=None,
            img_name=None,
            output_dir='heatmap',
            concept_set=None,
    ):
        pre_c = self.pre_concept_model(x)
        contexts = []
        c_sem = []

        for i, context_gen in enumerate(self.concept_context_generators):
            if self.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[i]
            context = context_gen(pre_c)
            prob = prob_gen(context)
            contexts.append(torch.unsqueeze(context, dim=1))
            c_sem.append(self.sigmoid(prob))
        c_sem = torch.cat(c_sem, dim=-1)
        contexts = torch.cat(contexts, dim=1)

        probs = c_sem
        # heatmap 可视化也复用统一的概念 embedding 组合逻辑，避免与 _forward 分叉。
        c_embedding = self.compute_concept_embedding(
            contexts[:, :, :self.emb_size],
            contexts[:, :, self.emb_size:],
            probs,
        )
        image_feature = self.unlabeled_image_encoder(x)

        heatmap = []
        for i in range(len(image_feature)):
            heatmap.append(torch.matmul(image_feature[i], c_embedding[i].transpose(0, 1)))
        heatmap = torch.stack(heatmap).permute(0, 3, 1, 2)

        return heatmap

        # print("heatmap")
        # print(heatmap.shape)
        # print(image_feature.shape)
        # print(c_embedding.shape)
        # exit(0)

        # for i in range(len(x)):
        #     # save_dir = f"./{output_dir}/{img_name[i]}"
        #     save_dir = f"/root/autodl-tmp/heatmap/{img_name[i]}"
        #
        #     visualize_and_save_heatmaps(x_show[i], c[i], c_sem[i], heatmap[i], save_dir, concept_set)
