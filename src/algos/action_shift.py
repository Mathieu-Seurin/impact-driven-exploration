import logging
import os
import threading
import time
import timeit
import pprint
import traceback

import torch
from torch import multiprocessing as mp
from torch import nn

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

from src.env_utils import FrameStack
from src.utils import get_batch, log, create_env, create_buffers, act
from src.action_hist import FlatHisto, WindowedHisto

#MinigridActionDistributionNet = models.MinigridActionDistribution
MinigridPolicyNet = models.MinigridPolicyNet
MinigridStateEmbeddingNet = models.MinigridStateEmbeddingNet
MinigridForwardDynamicsNet = models.MinigridForwardDynamicsNet
MinigridInverseDynamicsNet = models.MinigridInverseDynamicsNet


MarioDoomStateEmbeddingNet = models.MarioDoomStateEmbeddingNet
MarioDoomForwardDynamicsNet = models.MarioDoomForwardDynamicsNet
MarioDoomInverseDynamicsNet = models.MarioDoomInverseDynamicsNet
MarioDoomPolicyNet = models.MarioDoomPolicyNet

FullObsMinigridStateEmbeddingNet = models.FullObsMinigridStateEmbeddingNet
FullObsMinigridPolicyNet = models.FullObsMinigridPolicyNet


def learn(actor_model,
          model,
          action_hist,
          batch,
          initial_agent_state,
          optimizer,
          # action_distribution_optimizer,
          scheduler,
          flags,
          frames=None,
          state_embedding_model=None,
          forward_dynamics_model=None,
          inverse_dynamics_model=None,
          state_embedding_optimizer=None,
          forward_dynamics_optimizer=None,
          inverse_dynamics_optimizer=None,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        count_rewards = torch.ones((flags.unroll_length, flags.batch_size),
                                   dtype=torch.float32).to(device=flags.device)
        count_rewards = batch['episode_state_count'][1:].float().to(device=flags.device)

        action_id, count_action = torch.unique(batch["action"].flatten(), return_counts=True)

        if state_embedding_model:
            current_state_embedding = state_embedding_model(batch['partial_obs'][:-1].to(device=flags.device))
            next_state_embedding = state_embedding_model(batch['partial_obs'][1:].to(device=flags.device))

            batch_action_acted = torch.abs(current_state_embedding - next_state_embedding).sum(dim=2) > flags.change_treshold
            acted_id, action_acted = torch.unique((batch["action"][:-1] + 1) * batch_action_acted.long() - 1, return_counts=True)

            pred_next_state_emb = forward_dynamics_model(
                current_state_embedding, batch['action'][1:].to(device=flags.device))
            pred_actions = inverse_dynamics_model(current_state_embedding, next_state_embedding)

            forward_dynamics_loss = flags.forward_loss_coef * \
                                    losses.compute_forward_dynamics_loss(pred_next_state_emb, next_state_embedding)

            inverse_dynamics_loss = flags.inverse_loss_coef * \
                                    losses.compute_inverse_dynamics_loss(pred_actions, batch['action'][1:])


        else:
            acted_id, action_acted = torch.unique((batch["action"] + 1) * batch["action_acted"] - 1, return_counts=True)
            batch_action_acted = batch["action_acted"][:-1].byte()

            forward_dynamics_loss = torch.zeros(1)
            inverse_dynamics_loss = torch.zeros(1)

        action_hist.add(action_id, count_action.cpu().float(), acted_id[1:].cpu(), action_acted[1:].cpu().float())

        action_rewards = torch.zeros_like(batch["action"][:-1]).float()
        acted_ratio = action_hist.usage_ratio()

        if flags.action_dist_decay_coef == 0:
            reward_for_an_action = 1 - acted_ratio
        else:
            reward_for_an_action = torch.exp(- acted_ratio * flags.action_dist_decay_coef)
        reward_for_an_action[acted_ratio == 1] = 0

        reward_for_an_action[torch.isnan(reward_for_an_action)] = 0
        assert torch.all(reward_for_an_action >= 0), "Problem, reward should only be positive"


        for id in action_id:
            action_rewards[(batch["action"][:-1]==id.item()) & batch_action_acted] = reward_for_an_action[id]

        intrinsic_rewards = count_rewards * action_rewards

        intrinsic_reward_coef = flags.intrinsic_reward_coef
        intrinsic_rewards *= intrinsic_reward_coef


        learner_outputs, unused_state = model(batch, initial_agent_state)

        bootstrap_value = learner_outputs['baseline'][-1]

        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch['reward']
        if flags.no_reward:
            total_rewards = intrinsic_rewards
        else:
            total_rewards = rewards + intrinsic_rewards
        clipped_rewards = torch.clamp(total_rewards, -1, 1)

        discounts = (~batch['done']).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                                      batch['action'],
                                                      vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss #+ act_distrib_loss

        episode_returns = batch['episode_return'][batch['done']]
        stats = {
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mean_rewards': torch.mean(rewards).item(),
            'mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'mean_total_rewards': torch.mean(total_rewards).item(),
            'mean_action_rewards': torch.mean(action_rewards).item(),
            'mean_count_rewards': torch.mean(count_rewards).item(),
            'forward_dynamics_loss': forward_dynamics_loss.item(),
            'inverse_dynamics_loss': inverse_dynamics_loss.item(),
        }

        scheduler.step()
        optimizer.zero_grad()

        if state_embedding_model:
            state_embedding_optimizer.zero_grad()
            forward_dynamics_optimizer.zero_grad()
            inverse_dynamics_optimizer.zero_grad()

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)

        if state_embedding_model:
            nn.utils.clip_grad_norm_(state_embedding_model.parameters(), flags.max_grad_norm)
            nn.utils.clip_grad_norm_(forward_dynamics_model.parameters(), flags.max_grad_norm)
            nn.utils.clip_grad_norm_(inverse_dynamics_model.parameters(), flags.max_grad_norm)

        optimizer.step()

        if state_embedding_model:
            state_embedding_optimizer.step()
            forward_dynamics_optimizer.step()
            inverse_dynamics_optimizer.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def train(flags):
    if flags.xpid is None:
        flags.xpid = 'torchbeast-%s' % time.strftime('%Y%m%d-%H%M%S')
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(os.path.expanduser(
        '%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    env = create_env(flags)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)

    if 'MiniGrid' in flags.env:
        if flags.use_fullobs_policy:
            model = FullObsMinigridPolicyNet(env.observation_space.shape, env.action_space.n)
        else:
            model = MinigridPolicyNet(env.observation_space.shape, env.action_space.n)

        if flags.change_treshold >= 0:
            state_embedding_model = MinigridStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)
            forward_dynamics_model = MinigridForwardDynamicsNet(env.action_space.n).to(device=flags.device)
            inverse_dynamics_model = MinigridInverseDynamicsNet(env.action_space.n).to(device=flags.device)

        else:
            state_embedding_model = None
            forward_dynamics_model = None
            inverse_dynamics_model = None

    else:
        raise NotImplementedError("Doom is not available")
        # model = MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n)
        #
        # if flags.change_treshold >= 0:
        #     state_embedding_model = MarioDoomStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)
        # else:
        #     state_embedding_model = None

    if flags.histogram_length:
        action_hist = WindowedHisto(env.action_space.n, flags.histogram_length)
    else:
        action_hist = FlatHisto(env.action_space.n)

    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags)

    model.share_memory()

    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context('fork')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    episode_state_count_dict = dict()
    train_state_count_dict = dict()
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, buffers,
                  episode_state_count_dict, train_state_count_dict,
                  initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    if 'MiniGrid' in flags.env:
        if flags.use_fullobs_policy:
            learner_model = FullObsMinigridPolicyNet(env.observation_space.shape, env.action_space.n) \
                .to(device=flags.device)
        else:
            learner_model = MinigridPolicyNet(env.observation_space.shape, env.action_space.n) \
                .to(device=flags.device)
    else:
        learner_model = MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n) \
            .to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    if state_embedding_model:
        state_embedding_optimizer = torch.optim.RMSprop(
            state_embedding_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)

        inverse_dynamics_optimizer = torch.optim.RMSprop(
            inverse_dynamics_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)

        forward_dynamics_optimizer = torch.optim.RMSprop(
            forward_dynamics_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
    else:
        state_embedding_optimizer = None
        inverse_dynamics_optimizer = None
        forward_dynamics_optimizer = None


    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger('logfile')
    stat_keys = [
        'total_loss',
        'mean_episode_return',
        'pg_loss',
        'baseline_loss',
        'entropy_loss',
        'mean_rewards',
        'mean_intrinsic_rewards',
        'mean_total_rewards',
        'mean_action_rewards',
        'mean_count_rewards',
        'forward_dynamics_loss',
        'inverse_dynamics_loss'
    ]
    logger.info('# Step\t%s', '\t'.join(stat_keys))
    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state = get_batch(free_queue, full_queue, buffers,
                                           initial_agent_state_buffers, flags, timings)
            stats = learn(actor_model=model,
                          model=learner_model,
                          action_hist=action_hist,
                          batch=batch,
                          initial_agent_state=agent_state,
                          optimizer=optimizer, #action_distribution_optimizer,
                          scheduler=scheduler,
                          flags=flags,
                          frames=frames,
                          state_embedding_model=state_embedding_model,
                          forward_dynamics_model=forward_dynamics_model,
                          inverse_dynamics_model=inverse_dynamics_model,
                          state_embedding_optimizer=state_embedding_optimizer,
                          inverse_dynamics_optimizer=inverse_dynamics_optimizer,
                          forward_dynamics_optimizer=forward_dynamics_optimizer)

            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        checkpointpath = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                             'model.tar')))

        action_hist_path = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                             'action_hist.tar')))

        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
            #'action_distribution_optimizer_state_dict': action_distribution_optimizer.state_dict(),
            # 'action_distribution_model_state_dict': action_distribution_model.state_dict(),
        }, checkpointpath)

        try:
            action_hist_list = torch.load(action_hist_path)
        except FileNotFoundError:
            action_hist_list = []

        action_hist_list.append(action_hist.return_full_hist())
        torch.save(action_hist_list, action_hist_path)


    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:

            add_actor = []
            for actor_num, actor in enumerate(actor_processes):
                if not actor.is_alive():
                    actor.terminate()
                    new_actor = ctx.Process(
                        target=act,
                        args=(actor_num, free_queue, full_queue, model, buffers,
                              episode_state_count_dict, train_state_count_dict,
                              initial_agent_state_buffers, flags))
                    new_actor.start()
                    add_actor.append([actor_num, new_actor])

            for num, actor in add_actor:
                actor_processes[num] = actor

            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''
            total_loss = stats.get('total_loss', float('inf'))
            log.info('After %i frames: loss %f @ %.1f fps. %sStats:\n%s',
                     frames, total_loss, fps, mean_return,
                     pprint.pformat(stats))

            #memory_tracker.print_diff()
            # memory = pd.DataFrame(mem.create_summary(), columns=['object', 'number_of_objects', 'memory'])
            # memory['mem_per_object'] = memory['memory'] / memory['number_of_objects']
            # log.info("\n" + str(memory.sort_values('memory', ascending=False).head(10)))
            # log.info("\n" + str(memory.sort_values('mem_per_object', ascending=False).head(10)))

    except KeyboardInterrupt:
        return
    except Exception as e:
        log.error('Exception in main process')
        log.error(traceback.format_exc())
        raise e
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)  # When an actor receives None in its free_queue, it stops.
        for actor in actor_processes:
            actor.join(timeout=1)
    checkpoint(frames)
    plogger.close()

