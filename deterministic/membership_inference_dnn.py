import sys
import time
from math import ceil
from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn.functional as F

sys.path.append('../')

from torch.utils.data import DataLoader, Dataset

from common.datasets import GenericDataset
from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE


class MembershipInferenceAttack:
    def __init__(self, target_net: VanillaNetLinear, num_classes: int, distrib_moments: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.target_net = target_net
        self.target_net.eval()
        self.num_classes = num_classes
        self.shadow_models, self.attack_models = {}, {}
        for class_index in range(num_classes):
            self.shadow_models[class_index] = ShadowModel(num_classes, class_index, target_net.get_input_size()).to(TORCH_DEVICE)
            self.attack_models[class_index] = AttackModel(num_classes, class_index).to(TORCH_DEVICE)
        self.fwd_func = self.target_net.forward
        self.record_synthesizer = RecordSynthesizer(target_net, num_classes, distrib_moments, self.fwd_func)

    def train_shadow_models(self, batch_size: int, num_epochs: int, lr: float) -> Dict[int, Dataset]:
        input_per_attack_model = {}
        for class_index, shadow_model in self.shadow_models.items():
            shadow_model.train()
            # ------------ Hyperparameters ------------
            class_train_dset, class_test_dset = self.record_synthesizer.generate_training_data_for_class(class_index)
            shadow_train_loader = DataLoader(class_train_dset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(shadow_model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            # -----------------------------------------

            class_label = torch.tensor(class_index, dtype=torch.int).to(TORCH_DEVICE)
            batch_class_label = class_label.repeat(batch_size)
            for _ in range(num_epochs):
                for x, target in shadow_train_loader:
                    input_features, y = x.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
                    optimizer.zero_grad()
                    y_hat = shadow_model(input_features, batch_class_label)
                    loss = criterion(y_hat, y)
                    loss.backward()
                    optimizer.step()

            # At this point, the shadow model is trained
            # Do one more forward pass for both train and test to create the attack models dataset
            shadow_model.zero_grad()
            shadow_model.eval()
            shadow_test_loader = DataLoader(class_test_dset, batch_size=batch_size, shuffle=True)
            attack_model_x = torch.tensor([]).to(TORCH_DEVICE)
            attack_model_y = torch.tensor([]).to(TORCH_DEVICE)
            for data_train, _ in shadow_train_loader:
                x = data_train.to(TORCH_DEVICE)
                y_hat = shadow_model(x, batch_class_label)
                y_hat = F.softmax(y_hat, dim=1)
                attack_model_x = torch.cat((attack_model_x, y_hat), dim=0)
                attack_model_y = torch.cat((attack_model_y, torch.ones(batch_size, 1).to(TORCH_DEVICE)), dim=0)

            for data_test, _ in shadow_test_loader:
                x = data_test.to(TORCH_DEVICE)
                y_hat = shadow_model(x, batch_class_label)
                y_hat = F.softmax(y_hat, dim=1)
                attack_model_x = torch.cat((attack_model_x, y_hat), dim=0)
                attack_model_y = torch.cat((attack_model_y, torch.zeros(batch_size, 1).to(TORCH_DEVICE)), dim=0)

            input_per_attack_model[class_index] = GenericDataset(attack_model_x, attack_model_y)

        return input_per_attack_model


    def train_attack_models(self, input_per_attack_model: Dict[int, Dataset], batch_size: int, num_epochs: int, lr: float) -> None:
        for class_index, attack_model in self.attack_models.items():
            attack_model.train()
            # ------------ Hyperparameters ------------
            attack_train_loader = DataLoader(input_per_attack_model[class_index], batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
            criterion = torch.nn.BCEWithLogitsLoss()
            # -----------------------------------------

            for _ in range(num_epochs):
                for data, target in attack_train_loader:
                    #^ .detach() is absolutely necessary here, because the input is computed from the shadow models and hence
                    #^ without detaching they would remain in the computation graph and break when doing .backward()
                    x, y = data.detach().to(TORCH_DEVICE), target.detach().to(TORCH_DEVICE)
                    optimizer.zero_grad()
                    y_hat = attack_model(x)
                    loss = criterion(y_hat, y)
                    loss.backward()
                    optimizer.step()

    def test_attack_models(self, test_dset: Dataset) -> float:
        #* batch size of 1 for simplicity
        attack_test_loader = DataLoader(test_dset, batch_size=1, shuffle=True)
        for attack_model in self.attack_models.values():
            attack_model.eval()

        correct, total = 0, 0
        for data, target in attack_test_loader:
            x_test, y_test = data.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
            y_hat = self.fwd_func(x_test)
            predicted_label = torch.argmax(y_hat).tolist() # since predicted_label is a scalar tensor, tolist() gives a scalar
            attack_model_prediction = self.attack_models[predicted_label](y_hat)
            #* we can do this since the batch size is 1
            if torch.round(attack_model_prediction) == y_test[0]:
                correct += 1
            total += 1

        accuracy = correct / total
        print(f"Accuracy: {accuracy}")

        return accuracy

class AttackModel(torch.nn.Module):
    def __init__(self, num_classes, class_label) -> None:
        super(AttackModel, self).__init__()

        self.num_classes = num_classes
        # num_classes for prediction, num_classes for one-hot class label, 1 for shadow model "in" / "out"
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(self.num_classes, 1)
        )
        self.class_label = class_label

    def forward(self, inference_prediction: torch.Tensor) -> torch.Tensor:
        # inference_prediction is an unnormalised probability vector (i.e. logits), so first apply softmax
        softmaxed_inference_prediction = F.softmax(inference_prediction, dim=1)

        return self.seq(softmaxed_inference_prediction)

class ShadowModel(torch.nn.Module):
    def __init__(self, num_classes, class_label, num_input_features) -> None:
        super(ShadowModel, self).__init__()

        self.num_classes = num_classes
        self.class_label = class_label
        self.input_size = num_input_features + num_classes
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.input_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.input_size // 2, self.input_size // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.input_size // 4, self.num_classes),
        )
        self.base_one_hot = torch.eye(num_classes).to(TORCH_DEVICE)

    def forward(self, x: torch.Tensor, correct_label: torch.Tensor) -> torch.Tensor:
        # make it from integer to one-hot
        one_hot_correct_label = self.base_one_hot[correct_label]

        return self.seq(torch.cat((x, one_hot_correct_label), dim=1))

    @property
    def get_class_index(self):
        return self.class_label

class RecordSynthesizer:
    def __init__(self, net: VanillaNetLinear, num_classes: int, distrib_moments: Tuple[torch.Tensor, torch.Tensor],
                 fwd_func: Callable[[torch.Tensor], torch.Tensor], pos_training_samples: int = 7000) -> None:
        self.target_net = net
        self.marginal_means, self.marginal_stds = distrib_moments
        # Initialize all the shadow models
        self.num_classes = num_classes
        self.shadow_models_train_dsets = {}
        self.num_training_samples = pos_training_samples # per class
        #^ This is an argument to reduce the duplication when doing the synthetization on BNNs
        self.forward_func = fwd_func

    def generate_training_data_for_class(self, class_index) -> Tuple[Dataset, Dataset]:
        #^ See https://arxiv.org/pdf/1610.05820.pdf section V, part D
        # Generate the positive records for each class
        #* For each shadow model, we have the positive samples (named D_train_shadow_i in the paper), where i index of shadow model
        train_samples = self.__generate_class_synthetic_dataset(class_index)
        #* For the negative samples, we need a disjoint set of samples, named in the paper D_test_shadow_i
        #* The latter are not in the training set and will hence will be labeled as "out" (== 0)
        test_samples = self.__generate_disjoint(train_samples)
        # Combine the positive and negative samples
        targets = torch.eye(self.num_classes)[class_index].repeat(self.num_training_samples, 1).to(TORCH_DEVICE)
        targets.type(torch.LongTensor)
        train_dset = GenericDataset(train_samples, targets)
        test_dset = GenericDataset(test_samples, targets)

        return train_dset, test_dset

    def __generate_class_synthetic_dataset(self, class_index: int) -> torch.Tensor:
        generated_records = []
        start = time.time()
        while len(generated_records) < self.num_training_samples:
            record = self.__synthesize_record(class_index)
            if not isinstance(record, bool):
                generated_records.append(record.tolist())
        end = time.time()
        print(f"Generated {len(generated_records)} records for class {class_index}")
        print(f"Time taken: {end - start} s")

        return torch.tensor(generated_records)

    def __generate_disjoint(self, pos_samples: torch.Tensor) -> torch.Tensor:
        disjoint_dset_size = len(pos_samples)
        pos_set = set(pos_samples)
        neg_test_samples = []
        while len(neg_test_samples) < disjoint_dset_size:
            neg_record = torch.randn(self.target_net.get_input_size(), device=TORCH_DEVICE)
            if neg_record not in pos_set:
                neg_test_samples.append(neg_record.tolist())

        return torch.tensor(neg_test_samples)


    # ------------------------ Synthesize the data ------------------------
    def __synthesize_record(self, data_category: str, max_rand_features: int = 128, min_rand_features: int = 4,
                            max_it: int = 200, min_confidence: int = 0.2, max_rejections: int = 10) -> Union[torch.Tensor, bool]:
        #^ see https://arxiv.org/pdf/1610.05820.pdf section V, part C
        # data category is the class index
        # Since we have statistical moments, we can generate the data from a normal distribution with the same mean and std
        x = torch.normal(self.marginal_means.unsqueeze(0), self.marginal_stds.unsqueeze(0)).to(TORCH_DEVICE)
        y_category_threshold, j, k, curr_x = 0, 0, max_rand_features, x.detach().clone()

        # Define the conditions
        is_better = lambda y_hat: min_confidence < y_hat[data_category]
        is_prediction_correct = lambda y_hat: data_category == torch.argmax(y_hat)
        is_randomly_selected = lambda y_hat: torch.rand(1).to(TORCH_DEVICE) < y_hat[data_category]

        for _ in range(max_it):
            # [0] because we have a batch size of 1
            y_hat = F.softmax(self.forward_func(x)[0], dim=0)
            if y_category_threshold < y_hat[data_category]:
                if is_better(y_hat) and is_prediction_correct(y_hat) and is_randomly_selected(y_hat):
                    return x.squeeze(0)
                curr_x = x.detach().clone()
                y_category_threshold = y_hat[data_category]
                j = 0
            else:
                j += 1
                if j > max_rejections:
                    k = max(min_rand_features, ceil(k / 2))
                    j = 0
            # randomize k features from curr_x
            shuffled_indices = torch.randperm(self.target_net.get_input_size())[:k]
            x = curr_x.detach().clone()
            x[0, shuffled_indices] = torch.rand(k, device=TORCH_DEVICE, dtype=torch.float32)

        return False
