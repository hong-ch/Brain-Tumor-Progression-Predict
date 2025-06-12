
import pandas as pd
import numpy as np


# 파일명 직접 지정 (현재 폴더에 있다고 가정) 여기 경로를 지정해주세요.
csv_filename = 'data/patient_slices_data.csv'
meta_filename = 'data/DQNmetadata.csv'

# 2. 데이터 로딩
df = pd.read_csv(csv_filename)
meta = pd.read_excel(xlsx_filename)

# 3. 시계열 기본 feature 정의
base_feature_cols = ['mean', 'std', 'max', 'min', 'volume']
group_cols = ['patient_id', 'timepoint', 'category']
agg_dict = {col: 'mean' if col != 'volume' else 'sum' for col in base_feature_cols}
state_df = df.groupby(group_cols).agg(agg_dict).reset_index()

# 4. timepoint를 datetime으로 변환 및 정렬
state_df['timepoint'] = pd.to_datetime(state_df['timepoint'])
state_df = state_df.sort_values(['patient_id', 'timepoint'])

# 5. days_gap (경과일수) 파생변수 생성
state_df['days_gap'] = state_df.groupby('patient_id')['timepoint'].diff().dt.days.fillna(0).astype(int)
# (최초 시점은 0)

# 6. 시계열 feature 정규화 (base_feature_cols + days_gap)
all_time_features = base_feature_cols + ['days_gap']
feature_min = state_df[all_time_features].min()
feature_max = state_df[all_time_features].max()
state_df_norm = state_df.copy()
state_df_norm[all_time_features] = (state_df[all_time_features] - feature_min) / (feature_max - feature_min + 1e-8)

# 7. 메타데이터 전처리 (환자 id 제외 모두 feature로 사용)
meta_cols = [col for col in meta.columns if col != 'patient_id']
meta_use = meta.copy()

# 문자/범주형 feature는 label encoding, 숫자형은 정규화
for col in meta_cols:
    if meta_use[col].dtype == object or meta_use[col].dtype.name == 'category':
        meta_use[col] = meta_use[col].astype('category').cat.codes
for col in meta_cols:
    if pd.api.types.is_numeric_dtype(meta_use[col]):
        min_, max_ = meta_use[col].min(), meta_use[col].max()
        if max_ > min_:
            meta_use[col] = (meta_use[col] - min_) / (max_ - min_ + 1e-8)

# 8. 시계열+메타데이터 merge
state_df_merged = pd.merge(state_df_norm, meta_use, on='patient_id', how='left')

# 9. 최종 feature list: 시계열 + days_gap + 메타데이터 (날짜/정렬용은 반드시 빼기)
exclude_cols = ['timepoint', 'future_date']
feature_cols_full = []
for col in all_time_features + meta_cols:
    if col not in exclude_cols and col not in feature_cols_full:
        feature_cols_full.append(col)
print('최종 feature list:', feature_cols_full)

# 10. train episode 생성 (정렬만 timepoint, feature에는 넣지 않음)
train_states = state_df_merged[state_df_merged['category'] == 'train']
feature_cols_full = ['mean', 'std', 'max', 'min', 'volume', 'days_gap_x', 'sex', 'age', 'age_2', 'Weight_1', 'Weight_2']

patients_data = []
for pid, group in train_states.groupby('patient_id'):
    group = group.sort_values('timepoint_x')
    state_seq = group[feature_cols_full].values.astype(np.float32)
    volume_seq = group['volume'].values.astype(np.float32)
    if len(state_seq) < 2:
        continue
    patients_data.append({'patient_id': pid, 'states': state_seq, 'volumes': volume_seq})
print(f'총 train episode 개수: {len(patients_data)}')

# 2. 환경 클래스 (feature shape 자동 반영)

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

class BrainTumorAgentEnv(py_environment.PyEnvironment):
    def __init__(self, patients_data, obs_dim, delta=0.05, reward_tol=0.05):
        self.patients_data = patients_data
        self.n_actions = 3  # 0: 감소, 1: 유지, 2: 증가
        self.delta = delta
        self.reward_tol = reward_tol  # 정답 허용 오차
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2)
        self._observation_spec = array_spec.ArraySpec(
            shape=(obs_dim,), dtype=np.float32)  # feature 수 자동!
        self._max_steps = None
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.patient_idx = np.random.randint(len(self.patients_data))
        self.state_seq = self.patients_data[self.patient_idx]['states']
        self.volume_seq = self.patients_data[self.patient_idx]['volumes']
        self.t = 0
        self._episode_ended = False
        self._max_steps = len(self.state_seq)
        self.pred_volumes = [self.state_seq[0][4]]
        return ts.restart(self._get_obs())

    def _get_obs(self):
        obs = self.state_seq[self.t].copy()
        obs[4] = self.pred_volumes[-1]
        return obs

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        prev_pred_vol = self.pred_volumes[-1]
        action_effect = [-self.delta, 0.0, self.delta]  # 감소, 유지, 증가
        new_pred_vol = prev_pred_vol + action_effect[action]
        new_pred_vol = np.clip(new_pred_vol, 0.0, 1.0)
        self.pred_volumes.append(new_pred_vol)
        self.t += 1

        if self.t >= self._max_steps - 1:
            self._episode_ended = True
            true_last_volume = self.volume_seq[self.t]
            pred_last_volume = self.pred_volumes[-1]
            # ----- 교수님 방식: 오차가 일정 이하일 때만 +1, 아니면 0 -----
            if abs(pred_last_volume - true_last_volume) < self.reward_tol:
                reward = 1.0
            else:
                reward = 0.0
            return ts.termination(self._get_obs(), reward)
        else:
            return ts.transition(self._get_obs(), reward=0.0, discount=1.0)

# 3. DQN Q-Network & Agent 생성

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent

obs_dim = len(feature_cols_full)
py_env = BrainTumorAgentEnv(patients_data, obs_dim=obs_dim, delta=0.05)
tf_env = tf_py_environment.TFPyEnvironment(py_env)

q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(128, 128)
)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=tf.keras.losses.Huber(),
    train_step_counter=tf.Variable(0)
)
agent.initialize()

# 4. Replay Buffer / Policy / 학습 코드

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=5000
)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

# 초기 경험 쌓기
random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
for _ in range(200):
    collect_step(tf_env, random_policy, replay_buffer)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=32, num_steps=2).prefetch(3)
iterator = iter(dataset)

num_iterations = 5000
for step in range(num_iterations):
    collect_step(tf_env, agent.policy, replay_buffer)
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss
    if step % 500 == 0:
        print(f"Step {step}: loss = {train_loss.numpy():.4f}")

# 5. 검증/테스트 평가

val_states = state_df_merged[state_df_merged['category'] == 'val']
val_patients_data = []
for pid, group in val_states.groupby('patient_id'):
    group = group.sort_values('timepoint_x')   # <-- 여기!
    state_seq = group[feature_cols_full].values.astype(np.float32)
    volume_seq = group['volume'].values.astype(np.float32)
    if len(state_seq) < 2:
        continue
    val_patients_data.append({'patient_id': pid, 'states': state_seq, 'volumes': volume_seq})


eval_py_env = BrainTumorAgentEnv(val_patients_data, obs_dim=obs_dim, delta=0.05)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

num_eval_episodes = len(val_patients_data)
rewards = []
for _ in range(num_eval_episodes):
    time_step = eval_env.reset()
    total_reward = 0
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        total_reward += time_step.reward.numpy()
    rewards.append(total_reward)
print("평균 에피소드 reward (validation):", np.mean(rewards))


