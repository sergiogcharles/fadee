import numpy as np
import math
import torch

NUM_MOTORS = 12

# Generate the 12 angle constraints for a task
def generate_task(full_angle_of_motion, success_rate=0.8, num_expected_complete_failures=0):
    # how much limited motion there should be, e.g. success_rate=80% has 20% in expectation of failure
    mu = success_rate * full_angle_of_motion
    sigma = 1

    # keep sampling until in range 0 to full_angle_of_motion
    A = []
    for _ in range(12):
        while True:
            # Bernoulli: p = num_expected_complete_failures / num_motors, e.g. 2/12
            p = num_expected_complete_failures / 12
            fail = np.random.binomial(1, p)
            if fail:
               A.append(0)
               break 

            # Sample angle upper bound limit from normal
            angle_upper_bound = np.random.normal(mu, sigma)
            if 0 < angle_upper_bound <= full_angle_of_motion:
                A.append(angle_upper_bound)
                break

    ranges_of_motion = [(-a , a) for a in A]
    return ranges_of_motion

# generate meta train tasks
def generate_meta_train_tasks(num_meta_train, full_range_of_motion, num_motors=NUM_MOTORS):
    # T = torch.zeros((num_meta_train, num_motors))
    T = [[None] * num_motors] * num_meta_train

    for t in range(num_meta_train):
        # No expected complete failures
        task_t = generate_task(full_range_of_motion, success_rate=0.8, num_expected_complete_failures=0)
        T[t] = task_t

    return T

# generate meta test task
def generate_meta_test_task(full_range_of_motion, num_motors=NUM_MOTORS):
    T = [[None] * num_motors]

    # Out-of-distribution: success rate 10%, expected failures 3
    T[0] = generate_task(full_range_of_motion, success_rate=0.1, num_expected_complete_failures=3)

    return T    

if __name__ == '__main__':
    # print(generate_task(np.pi / 4))
    meta_train_tasks_ranges = generate_meta_train_tasks(1000, math.pi / 4)
    meta_test_task_ranges = generate_meta_test_task(math.pi / 4)

    print(meta_train_tasks_ranges)
    print(meta_test_task_ranges)




