from jobflow_remote.jobs.jobcontroller import JobController
import re
from datetime import datetime
from datetime import datetime
import numpy as np
def time_diff(t1: str, t2: str, unit: str = "seconds") -> float:
    delta = datetime.fromisoformat(t2) - datetime.fromisoformat(t1)
    seconds = delta.total_seconds()

    if unit == "seconds":
        return seconds
    elif unit == "minutes":
        return seconds / 60
    elif unit == "hours":
        return seconds / 3600
    else:
        raise ValueError("unit must be 'seconds', 'minutes', or 'hours'")

jc = JobController.from_project_name("autoplex_project")
database_ids = [f.db_ids for f in jc.get_flows_info()]

num_completed=0
run_time=[]
jc.get_jobs_info(db_ids=database_ids[-1])
for id in database_ids[-1]:
    if str(jc.get_job_info(db_id=id).state)=="JobState.COMPLETED":
        num_completed+=1
        run_time.append(time_diff(
            str(jc.get_job_info(db_id=id).start_time),
            str(jc.get_job_info(db_id=id).end_time),
            unit="hours")
        )

print(num_completed)
#print(run_time)
print(np.mean(run_time))
