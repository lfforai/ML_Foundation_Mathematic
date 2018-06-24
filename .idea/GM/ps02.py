import argparse
import sys

import tensorflow as tf

FLAGS = None


def main(_):
    print(FLAGS)
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        pass



if __name__ == "__main__":
    tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
    tf.app.flags.DEFINE_string('ps_hosts', '',
                               """Comma-separated list of hostname:port for the """
                               """parameter server jobs. e.g. """
                               """'machine1:2222,machine2:1111,machine2:2222'""")
    tf.app.flags.DEFINE_string('worker_hosts', '',
                               """Comma-separated list of hostname:port for the """
                               """worker jobs. e.g. """
                               """'machine1:2222,machine2:1111,machine2:2222'""")
    tf.app.flags.DEFINE_integer(
        'task_id', 0, 'Task id of the replica running the training.')
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)