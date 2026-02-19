mongod --bind_ip_all --dbpath /u/vld/sedm7085/DB/ --quiet --port 2000

Description

The states can then be grouped in

    Waiting states: describing Jobs that has not started yet.

    Running states: states for which the Runner has started working on the Job

    Completed state: a state where the Job has been completed successfully

    Error states: states associated with some error in the Job, either programmatic or during the execution.

The list of Job states is defined in the jobflow_remote.jobs.state.JobState object. Here we present a list of each state with a short description.
WAITING

Waiting state. A Job that has been inserted into the database but has to wait for other Jobs to be completed before starting.
READY

Waiting state. A Job that is ready to be executed by the Runner.
CHECKED_OUT

Running state. A Job that has been selected by the Runner to start its execution.
UPLOADED

Running state. All the inputs required by the Job has been copied to the worker.
SUBMITTED

Running state. The Job has been submitted to the queueing system of the worker.
RUNNING

Running state. The Runner verified that the Job has started is being executed on the worker.
TERMINATED

Running state. The process executing the Job on the worked has finished running. No knowledge of whether this happened for an error or because the Job was completed correctly is available at this point.
DOWNLOADED

Running state. The Runner has copied to the local machine all the files containing the Job response and outputs to be stored.
COMPLETED

Completed state. A Job that has completed correctly.
FAILED

Error state. The procedure to execute the Job completed correctly, but an error happened during the execution of the Job’s function, so the Job did not complete successfully.
REMOTE_ERROR

Error state. An error occurred during the procedure to execute the Job. For example the files could not be copied due to some network error and the maximum number of attempts has been reached. The Job may or may not be executed, depending on the action that generated the issue, but in any case no information is available about it. This failure is independent from the correct execution of the Job’s function.
PAUSED

Waiting state. The Job has been paused by the user and will not be executed by the Runner. A Job in this state can be started again.
STOPPED

Error state. The Job was stopped by another Job as a consequence of a stop_jobflow or stop_children actions in the Job’s response. A Job in this state can be resumed.
USER_STOPPED

Error state. A Job stopped by the user. A Job in this state can be resumed.
BATCH_SUBMITTED

Running state. A Job submitted for execution to a batch worker. Differs from the SUBMITTED state since the Runner does not have to check its state in the queueing system.
BATCH_RUNNING

Running state. A Job that is being executed by a batch worker. Differs from the RUNNING state since the Runner does not have to check its state in the queueing system.
Evolution

If the state of a Job is not directly modified by user, the Runner will consistently update the state of each Job in a running state.

The following diagram illustrates which states transitions can be performed by the Runner on a Job. This includes the transitions to intermediate or final error states.

module load cuda/12.1

# Jobflow commands

jf job info [id]
jf flow list
jf flow delete -did [id]
jf flow info [id]
jf runner start
jf runner kill
jf project check --errors
jf job list -s REMOTE_ERROR
jf job list -s RUNNING
jf job list -s WAITING
jf job list -s FAILED
jf job list -n machine_learning_fit
jf job rerun [id]

tmux #creats a virtual terminal that persists  MUST BE USED IN (base)
Ctrl + b, then d to exit but keep alive
tmux list-sessions or tmux ls 

gunzip *.gz # for unzipping .gz castep output files
find . -type f -name "*.gz" -exec gunzip {} + #unzip all files from relaxation folder

shift+PrtScn to screenshot a small area.