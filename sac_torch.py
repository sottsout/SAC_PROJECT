# import os
# import torch as T
# import torch.nn.functional as F
# import numpy as np
# from buffer import ReplayBuffer
# from networks import ActorNetwork, CriticNetwork, ValueNetwork
# from pathlib import Path
#
# class Agent():
#     def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
#                  env=None, gamma=0.99, n_actions=2, max_size=2_000_000, tau=0.005,
#                  layer1_size=256, layer2_size=256, batch_size=256, reward_scale=3 ,automatic_entropy_tuning = True):
#
#         self.learn_step_cntr = 0
#         self.target_update_interval = 2
#         self.gamma = gamma#bellman parameter
#         self.tau = tau#ΓΙΑ ΝΑ ΚΑΝΟΥΜΕ SOFT UPDATE
#         self.memory = ReplayBuffer(max_size, input_dims, n_actions)
#         self.batch_size = batch_size
#         self.n_actions = n_actions##### ΘΑ ΕΙΝΑΙ ΟΣΟ ΚΑΙ ΟΙ ΒΑΘΜΟΙ ΕΛΕΥΘΕΡΙΑΣ ΤΟΥ ΡΟΜΠΟΤ
#         self.chkpt_dir = Path("tmp/sac")  # centralized path
#         self.chkpt_dir.mkdir(parents=True, exist_ok=True)
#         self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,#ΟΡΙΖΟΥΜΕ ΤΟΝ ΑΚΤΟΡ - ΓΙΑ ΤΗΝ ΠΟΛΙΤΙΚΗ
#                                   name='actor', max_action=env.action_space.high)#ΟΡΙΖΟΥΜΕ ΤΟΝ ΑΚΤΟΡ - ΓΙΑ ΤΗΝ ΠΟΛΙΤΙΚΗ
#
#         self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,#ΟΡΙΖΟΥΜΕ ΤΑ 2 ΔΙΚΤΥΑ ΚΡΙΤΩΝ
#                                       name='critic_1')
#         self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,#ΟΡΙΖΟΥΜΕ ΤΑ 2 ΔΙΚΤΥΑ ΚΡΙΤΩΝ
#                                       name='critic_2')
#         #self.value = ValueNetwork(beta, input_dims, name='value')
#         #self.target_value = ValueNetwork(beta, input_dims, name='target_value')
#         self.target_critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='target_critic_1')
#         self.target_critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='target_critic_2')
#
#         #self.scale = reward_scale
#         self.update_network_parameters(tau=1)
#         self.automatic_entropy_tuning = automatic_entropy_tuning
#
#         if automatic_entropy_tuning:
#
#             #self.target_entropy = -np.prod(n_actions)
#             self.target_entropy = -env.action_space.shape[0]
#             self.log_alpha = T.zeros(1,requires_grad=True,device = self.actor.device)
#             self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha)
#         else:
#             self.alpha = 1.0
#
#     def choose_action(self, observation):#
#         state = T.Tensor([observation]).to(self.actor.device)# ΜΕΤΑΤΡΕΠΟΥΜΕ ΣΕ ΤΕΝΣΟΡΕΣ ΚΑΙ ΤΟ ΣΤΕΛΝΟΥΜΕ ΣΤΟ ACTOR . DEVICE
#         actions, _ = self.actor.sample_normal(state, reparameterize=False)
#
#         return actions.cpu().detach().numpy()[0]
#
#     def remember(self, state, action, reward, new_state, done):
#         self.memory.store_transition(state, action, reward, new_state, done)
#
#     # def update_network_parameters(self, tau=None):
#     #     if tau is None:
#     #         tau = self.tau
#     #
#     #     target_value_params = self.target_value.named_parameters()# ΠΑΙΡΝΕΙ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ -ΜΕΡΟΛΗΨΙΩΝ ΤΩΝ ΕΣΩΤΕΡΙΚΩΝ ΕΠΙΠΕΔΩΝ
#     #     value_params = self.value.named_parameters()# ΠΑΙΡΝΕΙ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ -ΜΕΡΟΛΗΨΙΩΝ ΤΩΝ ΕΣΩΤΕΡΙΚΩΝ ΕΠΙΠΕΔΩΝ
#     #
#     #     target_value_state_dict = dict(target_value_params)#ΦΤΙΑΧΝΕΙ ΛΕΞΙΚΟ ΤΟ ΟΠΟΙΟ ΘΑ ΕΧΕΙ ΩΣ ΚΛΕΙΔΙΑ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ /ΜΕΡΟΛΗΨΙΩΝ
#     #     value_state_dict = dict(value_params)#ΦΤΙΑΧΝΕΙ ΛΕΞΙΚΟ ΤΟ ΟΠΟΙΟ ΘΑ ΕΧΕΙ ΩΣ ΚΛΕΙΔΙΑ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ /ΜΕΡΟΛΗΨΙΩΝ
#     #
#     #     for name in value_state_dict:
#     #         value_state_dict[name] = tau * value_state_dict[name].clone() + \
#     #                                  (1 - tau) * target_value_state_dict[name].clone()###SOFT UPDATE ΤΩΝ ΠΑΡΑΜΕΤΡΩΝ
#     #
#     #     self.target_value.load_state_dict(value_state_dict)
#     def update_network_parameters(self, tau=None):
#         if tau is None:
#             tau = self.tau
#
#         for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
#             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
#
#         for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
#             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
#
#     def save_models(self):
#         print('.... saving models ....')
#         self.actor.save_checkpoint()
#         #self.value.save_checkpoint()
#        # self.target_value.save_checkpoint()
#         self.critic_1.save_checkpoint()
#         self.critic_2.save_checkpoint()
#         if self.automatic_entropy_tuning:
#             T.save(self.log_alpha, self.chkpt_dir / 'log_alpha')
#
#     def load_models(self):
#         print('.... loading models ....')
#         self.actor.load_checkpoint()
#        # self.value.load_checkpoint()
#        # self.target_value.load_checkpoint()
#         self.critic_1.load_checkpoint()
#         self.critic_2.load_checkpoint()
#         if self.automatic_entropy_tuning:
#             self.log_alpha = T.load(self.chkpt_dir / 'log_alpha')
#
#     def learn(self, gradient_steps: int = 20):
#         """
#         Εκτελεί `gradient_steps` ενημερώσεις κριτών/actor.
#         Κάνει Polyak update των target κριτών κάθε `target_update_interval` βήματα.
#         """
#         # ------------------------------------------------------------------
#         # 0) Warm-up: περιμένουμε να γεμίσει λίγο το replay buffer
#         # ------------------------------------------------------------------
#        #if self.memory.mem_cntr < max(self.batch_size, 1_000):
#        #    return
#
#         if self.memory.mem_cntr < max(10 * self.batch_size, 50_000):
#             return
#         # Δημιουργούμε counter & interval αν δεν υπάρχουν (ώστε να μη χρειάζονται αλλαγές στο __init__)
#         if not hasattr(self, "learn_step_cntr"):
#             self.learn_step_cntr = 0
#         if not hasattr(self, "target_update_interval"):
#             self.target_update_interval = 2        # π.χ. Polyak update κάθε 2 learn calls
#
#         # ------------------------------------------------------------------
#         # 1) Εκτελούμε N gradient steps
#         # ------------------------------------------------------------------
#         for _ in range(gradient_steps):
#
#             # Δείγμα minibatch από replay buffer
#             state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
#             state      = T.tensor(state,      dtype=T.float, device=self.actor.device)
#             action     = T.tensor(action,     dtype=T.float, device=self.actor.device)
#             reward     = T.tensor(reward,     dtype=T.float, device=self.actor.device).view(-1)
#             next_state = T.tensor(next_state, dtype=T.float, device=self.actor.device)
#             done       = T.tensor(done,       dtype=T.float, device=self.actor.device).view(-1)
#
#             # --------------------------------------------------------------
#             # 2) Υπολογισμός target-Q (χωρίς gradient)
#             # --------------------------------------------------------------
#             with T.no_grad():
#                 next_action, next_logp = self.actor.sample_normal(next_state, reparameterize=False)
#                 q1_next = self.target_critic_1(next_state, next_action).view(-1)
#                 q2_next = self.target_critic_2(next_state, next_action).view(-1)
#                 q_next = T.min(q1_next, q2_next)
#
#                 alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else T.tensor(
#                     self.alpha, device=self.actor.device
#                 )
#                 target_q = reward + self.gamma * (1 - done) * (q_next - alpha * next_logp.view(-1))
#
#             # --------------------------------------------------------------
#             # 3) Ενημέρωση κριτών
#             # --------------------------------------------------------------
#             self.critic_1.optimizer.zero_grad()
#             self.critic_2.optimizer.zero_grad()
#             current_q1 = self.critic_1(state, action).view(-1)
#             current_q2 = self.critic_2(state, action).view(-1)
#             critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
#             critic_loss.backward()
#             T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
#             T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
#             self.critic_1.optimizer.step()
#             self.critic_2.optimizer.step()
#
#             # --------------------------------------------------------------
#             # 4) Ενημέρωση actor
#             # --------------------------------------------------------------
#             self.actor.optimizer.zero_grad()
#             new_actions, logp = self.actor.sample_normal(state, reparameterize=True)
#             q1_pi = self.critic_1(state, new_actions)
#             q2_pi = self.critic_2(state, new_actions)
#             q_pi  = T.min(q1_pi, q2_pi).view(-1)
#             actor_loss = (alpha * logp.view(-1) - q_pi).mean()
#             actor_loss.backward()
#             T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
#             self.actor.optimizer.step()
#
#             # --------------------------------------------------------------
#             # 5) Ενημέρωση entropy coefficient (α)
#             # --------------------------------------------------------------
#             if self.automatic_entropy_tuning:
#                 alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
#                 self.alpha_optimizer.zero_grad()
#                 alpha_loss.backward()
#                 self.alpha_optimizer.step()
#
#                 if self.memory.mem_cntr % 1_000 == 0 and gradient_steps == 9:
#                     print(f"[α] entropy coeff: {self.log_alpha.exp().item():.4f}")
#
#             # --------------------------------------------------------------
#             # 6) Polyak update (κάθε N learn calls)
#             # --------------------------------------------------------------
#             self.learn_step_cntr += 1
#             if self.learn_step_cntr % self.target_update_interval == 0:
#                 self.update_network_parameters()
#
#
#

import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from pathlib import Path

class Agent():
    def __init__(self, alpha=0.00015, beta=0.00015, input_dims=[8],
                 env=None, gamma=0.95, n_actions=2, max_size=1_000_000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=512, reward_scale=2 ,automatic_entropy_tuning = True,logger = None):

        self.learn_step_cntr = 0
        self.target_update_interval = 2
        self.gamma = gamma#bellman parameter
        self.tau = tau#ΓΙΑ ΝΑ ΚΑΝΟΥΜΕ SOFT UPDATE
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions##### ΘΑ ΕΙΝΑΙ ΟΣΟ ΚΑΙ ΟΙ ΒΑΘΜΟΙ ΕΛΕΥΘΕΡΙΑΣ ΤΟΥ ΡΟΜΠΟΤ
        self.chkpt_dir = Path("tmp/sac")  # centralized path
        self.chkpt_dir.mkdir(parents=True, exist_ok=True)
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,#ΟΡΙΖΟΥΜΕ ΤΟΝ ΑΚΤΟΡ - ΓΙΑ ΤΗΝ ΠΟΛΙΤΙΚΗ
                                  name='actor', max_action=env.action_space.high)#ΟΡΙΖΟΥΜΕ ΤΟΝ ΑΚΤΟΡ - ΓΙΑ ΤΗΝ ΠΟΛΙΤΙΚΗ

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,#ΟΡΙΖΟΥΜΕ ΤΑ 2 ΔΙΚΤΥΑ ΚΡΙΤΩΝ
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,#ΟΡΙΖΟΥΜΕ ΤΑ 2 ΔΙΚΤΥΑ ΚΡΙΤΩΝ
                                      name='critic_2')
        #self.value = ValueNetwork(beta, input_dims, name='value')
        #self.target_value = ValueNetwork(beta, input_dims, name='target_value')
        self.target_critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='target_critic_2')
        self.logger = logger
        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.log_interval = 10000  # γράφε log κάθε 1000 learn steps

        if automatic_entropy_tuning:

            self.target_entropy = -np.prod(env.action_space.shape) * 0.98
           # self.target_entropy = -n_actions
            self.log_alpha = T.zeros(1,requires_grad=True,device = self.actor.device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha*0.1)
        else:
            self.alpha = 1.0

    def choose_action(self, observation):#
        state = T.Tensor([observation]).to(self.actor.device)# ΜΕΤΑΤΡΕΠΟΥΜΕ ΣΕ ΤΕΝΣΟΡΕΣ ΚΑΙ ΤΟ ΣΤΕΛΝΟΥΜΕ ΣΤΟ ACTOR . DEVICE
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # def update_network_parameters(self, tau=None):
    #     if tau is None:
    #         tau = self.tau
    #
    #     target_value_params = self.target_value.named_parameters()# ΠΑΙΡΝΕΙ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ -ΜΕΡΟΛΗΨΙΩΝ ΤΩΝ ΕΣΩΤΕΡΙΚΩΝ ΕΠΙΠΕΔΩΝ
    #     value_params = self.value.named_parameters()# ΠΑΙΡΝΕΙ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ -ΜΕΡΟΛΗΨΙΩΝ ΤΩΝ ΕΣΩΤΕΡΙΚΩΝ ΕΠΙΠΕΔΩΝ
    #
    #     target_value_state_dict = dict(target_value_params)#ΦΤΙΑΧΝΕΙ ΛΕΞΙΚΟ ΤΟ ΟΠΟΙΟ ΘΑ ΕΧΕΙ ΩΣ ΚΛΕΙΔΙΑ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ /ΜΕΡΟΛΗΨΙΩΝ
    #     value_state_dict = dict(value_params)#ΦΤΙΑΧΝΕΙ ΛΕΞΙΚΟ ΤΟ ΟΠΟΙΟ ΘΑ ΕΧΕΙ ΩΣ ΚΛΕΙΔΙΑ ΤΑ ΟΝΟΜΑΤΑ ΤΩΝ ΒΑΡΩΝ /ΜΕΡΟΛΗΨΙΩΝ
    #
    #     for name in value_state_dict:
    #         value_state_dict[name] = tau * value_state_dict[name].clone() + \
    #                                  (1 - tau) * target_value_state_dict[name].clone()###SOFT UPDATE ΤΩΝ ΠΑΡΑΜΕΤΡΩΝ
    #
    #     self.target_value.load_state_dict(value_state_dict)
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        #self.value.save_checkpoint()
       # self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        if self.automatic_entropy_tuning:
            T.save(self.log_alpha, self.chkpt_dir / 'log_alpha')

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
       # self.value.load_checkpoint()
       # self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        if self.automatic_entropy_tuning:
            self.log_alpha = T.load(self.chkpt_dir / 'log_alpha')

    def learn(self, gradient_steps: int = 20):
        """
        Εκτελεί `gradient_steps` ενημερώσεις κριτών/actor.
        Κάνει Polyak update των target κριτών κάθε `target_update_interval` βήματα.
        """
        # ------------------------------------------------------------------
        # 0) Warm-up: αν δεν έχει αρκετά samples στο replay buffer
        # ------------------------------------------------------------------
        if self.memory.mem_cntr < max(self.batch_size, 1_000):
            return 0.0, 0.0, 0.0, 0.0  # γύρνα μηδενικά losses στο warm-up

        if not hasattr(self, "learn_step_cntr"):
            self.learn_step_cntr = 0
        if not hasattr(self, "target_update_interval"):
            self.target_update_interval = 2

        # ------------------------------------------------------------------
        # 1) Gradient steps
        # ------------------------------------------------------------------
        for _ in range(gradient_steps):

            # δειγματοληψία minibatch
            state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
            state = T.tensor(state, dtype=T.float, device=self.actor.device)
            action = T.tensor(action, dtype=T.float, device=self.actor.device)
            reward = T.tensor(reward, dtype=T.float, device=self.actor.device).view(-1)
            next_state = T.tensor(next_state, dtype=T.float, device=self.actor.device)
            done = T.tensor(done, dtype=T.float, device=self.actor.device).view(-1)

            # 2) Υπολογισμός στόχου Q χωρίς gradient
            with T.no_grad():
                next_action, next_logp = self.actor.sample_normal(next_state, reparameterize=False)
                q1_next = self.target_critic_1(next_state, next_action).view(-1)
                q2_next = self.target_critic_2(next_state, next_action).view(-1)
                q_next = T.min(q1_next, q2_next)

                alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else T.tensor(
                    self.alpha, device=self.actor.device
                )
                target_q = reward + self.gamma * (1 - done) * (q_next - alpha * next_logp.view(-1))

            # 3) Update Critic networks
            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            current_q1 = self.critic_1(state, action).view(-1)
            current_q2 = self.critic_2(state, action).view(-1)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            critic_loss.backward()
            T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
            T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # 4) Update Actor network
            self.actor.optimizer.zero_grad()
            new_actions, logp = self.actor.sample_normal(state, reparameterize=True)
            q1_pi = self.critic_1(state, new_actions)
            q2_pi = self.critic_2(state, new_actions)
            q_pi = T.min(q1_pi, q2_pi).view(-1)
            actor_loss = (alpha * logp.view(-1) - q_pi).mean()
            actor_loss.backward()
            self.actor.optimizer.step()

            # 5) Update Entropy Coefficient (α)
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            else:
                alpha_loss = T.tensor(0.0, device=self.actor.device)

            # 6) Polyak update στα target networks
            self.learn_step_cntr += 1
            if self.learn_step_cntr % self.target_update_interval == 0:
                self.update_network_parameters()

        # ------------------------------------------------------------------
        # 2) Logging
        # ------------------------------------------------------------------
        ent_coef = self.log_alpha.exp().item() if self.automatic_entropy_tuning else self.alpha

        if self.logger is not None:
            self.logger.record("train/critic_loss", critic_loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.logger.record("train/ent_coef", ent_coef)
            if self.automatic_entropy_tuning:
                self.logger.record("train/alpha_loss", alpha_loss.item())

            # ➔ Κάνε dump μόνο κάθε log_interval steps
            if self.learn_step_cntr % self.log_interval == 0:
                self.logger.dump(step=self.learn_step_cntr)

        return critic_loss.item(), actor_loss.item(), ent_coef, (
            alpha_loss.item() if self.automatic_entropy_tuning else 0.0
        )



