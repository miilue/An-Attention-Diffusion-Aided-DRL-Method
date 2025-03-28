from .config import *


class ServiceProvider:

    def __init__(self, sid, eid):
        self._sid = sid
        self._eid = eid
        # self._task_type = task_type_id
        self._reward_baselines = [
            np.random.choice(TYPE1_RANGE),
            np.random.choice(TYPE2_RANGE),
            np.random.choice(TYPE3_RANGE),
            np.random.choice(TYPE4_RANGE)]
        print('-'*20)
        print('sid: ', sid)
        print('reward_baselines: ', self._reward_baselines)
        self._reward_coefs = (
            np.random.choice(AX_RANGE),
            np.random.choice(AY_RANGE),
            np.random.choice(BX_RANGE),
            np.random.choice(BY_RANGE))
        self._serving_tasks = []
        self._terminated_tasks = {'crashed': [], 'finished': []}
        print('reward_coefs: ', self._reward_coefs)
        print('-'*20)

        # # The following info are not currently considered
        # self._loc = np.random.randint(*LOCATION_RANGE, size=(1, 2))
        # self._num_cpu = NUM_CPUS
        # self._num_gpu = NUM_GPUS
        # self._cpu_mem = CPU_MEM
        # self._gpu_mem = GPU_MEM

    # def _distance_to(self, user):
    #     return np.sqrt(np.square(self._loc - user._loc).sum())

    @property
    def id(self):
        return self._sid

    @property
    def used_t(self):
        return sum([task.t for task in self._serving_tasks])

    def check_finished(self, curr_time):
        num_finished = 0
        for running_task_ in self._serving_tasks[:]:
            if running_task_.can_finished(curr_time):
                running_task_.set_finished()
                self._terminated_tasks['finished'].append(running_task_)
                self._serving_tasks.remove(running_task_)
                num_finished += 1
        return num_finished

    def calculate_reward(self, task):
        # reward will be delayed in practice
        return BETA * self._reward_baselines[task.task_type] + REWARD(*self._reward_coefs, task.t)

    def crashed_occur(self, curr_time):
        penalty = 0
        for running_task_ in self._serving_tasks:
            running_task_.crash(curr_time)
            self._terminated_tasks['crashed'].append(running_task_)
            penalty += (1 - running_task_.progress()) * CRASH_PENALTY_COEF
        self._serving_tasks.clear()
        return penalty

    def assign_task(self, task):
        reward = self.calculate_reward(task)
        self._serving_tasks.append(task)
        return reward

    def reset(self):
        self._serving_tasks.clear()
        self._terminated_tasks['crashed'].clear()
        self._terminated_tasks['finished'].clear()

    def task_summary(self):
        num_serving = len(self._serving_tasks)  # task_serving
        num_crashed = len(self._terminated_tasks['crashed'])  # task_crashed
        num_finished = len(self._terminated_tasks['finished'])  # task_finished
        crashed_total_t = sum(task.t for task in self._terminated_tasks['crashed'])
        crashed_total_reward = sum(self.calculate_reward(task) for task in self._terminated_tasks['crashed'])
        finished_total_t = sum(task.t for task in self._terminated_tasks['finished'])
        finished_total_reward = sum(self.calculate_reward(task) for task in self._terminated_tasks['finished'])
        return {
            "total": num_serving + num_crashed + num_finished,
            "serving": num_serving,
            "crashed": num_crashed,
            "finished": num_finished,
            "crashed_total_t": crashed_total_t,
            "crashed_total_reward": crashed_total_reward,
            "finished_total_t": finished_total_t,
            "finished_total_reward": finished_total_reward
        }

    @property
    def info(self):
        return {
            'id': self.id,
            'eid': self._eid,
            'task_serving': len(self._serving_tasks),
            'task_finished': len(self._terminated_tasks['finished']),
            'task_crashed': len(self._terminated_tasks['crashed']),
            'used_t': self.used_t
        }


class EdgeServer:
    def __init__(self, eid):
        self._eid = eid
        self._n_service_providers = NUM_SERVICE_PROVIDERS
        self._service_providers = [ServiceProvider(sid, eid) for sid in range(self._n_service_providers)]
        self._total_t = np.random.choice(TOTAL_T_RANGE)
        self.assign_sid = None
        self._num_crashed = 0
        print('-'*20)
        print('eid: ', eid)
        print('total_t: ', self._total_t)
        print('-'*20)

    @property
    def id(self):
        return self._eid

    @property
    def total_t(self):
        return self._total_t

    @property
    def used_t_sum(self):
        return sum([service.used_t for service in self._service_providers])

    @property
    def available_t(self):
        return self._total_t - self.used_t_sum

    def is_enough(self, task):
        return self.available_t >= task.t

    @property
    def norm_total_t(self):
        max_t = TOTAL_T_RANGE[-1]
        return self._total_t / max_t

    @property
    def norm_available_t(self):
        return self.available_t / self._total_t

    def check_finished(self, curr_time):
        num_finished_sum = 0
        for service in self._service_providers:
            num_finished_sum += service.check_finished(curr_time)
        return num_finished_sum

    def assign_task(self, task, curr_time):
        # No enough resources, server crashes
        if task.t > self.available_t:
            penalty = CRASH_PENALTY_COEF  # fixed penalty
            for service in self._service_providers:
                penalty += service.crashed_occur(curr_time)
            self._num_crashed += 1
            return -penalty
        else:
            reward = self._service_providers[self.assign_sid].assign_task(task)
            return reward

    def reset(self):
        for service in self._service_providers:
            service.reset()
        self.assign_sid = None
        self._num_crashed = 0

    # s^A_i
    @property
    def vector(self):
        # (total_t, available_t)
        return np.hstack([self.norm_total_t, self.norm_available_t, self._n_service_providers])

    def task_summary(self):
        summarys = []
        for service in self._service_providers:
            summarys.append(service.task_summary())
        num_serving = sum(summary['serving'] for summary in summarys)
        num_crashed = sum(summary['crashed'] for summary in summarys)
        num_finished = sum(summary['finished'] for summary in summarys)
        crashed_total_t = sum(summary['crashed_total_t'] for summary in summarys)
        crashed_total_reward = sum(summary['crashed_total_reward'] for summary in summarys)
        finished_total_t = sum(summary['finished_total_t'] for summary in summarys)
        finished_total_reward = sum(summary['finished_total_reward'] for summary in summarys)
        return {
            "total": num_serving + num_crashed + num_finished,
            "serving": num_serving,
            "crashed": num_crashed,
            "finished": num_finished,
            "crashed_total_t": crashed_total_t,
            "crashed_total_reward": crashed_total_reward,
            "finished_total_t": finished_total_t,
            "finished_total_reward": finished_total_reward
        }

    @property
    def info(self):
        infos = []
        for service in self._service_providers:
            service_info = service.info
            service_info['total_t'] = self._total_t
            service_info['available_t'] = self.available_t
            service_info['num_crashed'] = self._num_crashed
            infos.append(service_info)
        return infos
