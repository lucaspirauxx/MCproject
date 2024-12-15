import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math


def calendar_transition(token1,token2,token3,current_time):
    calendar ={}

    if token3=="D":# event sunset
        if token1=="F1" and token2=="S2" and (current_time %24)>15:
            calendar["n_ot_t"]=current_time + 1 # overtime transition
        elif 8<=(current_time %24)<=16:
            calendar["n_t"]=current_time + 16 - (current_time %24) 

    elif token3=="N":#event sunrise
        if token2=="W2":
            calendar["s2_t"]=current_time
        elif (current_time %24)>=16:
            calendar["d_t"]=current_time+ 32 - (current_time %24)
        else :
            calendar["d_t"]=current_time+ 8 - (current_time %24) 

    if token1=="W1"and token2=="W2":
        calendar["s2_t"]= current_time
    elif token1=="F1" and token2=="S2" and token3=="D":
        calendar["g1_t"]=(repair_time)+ current_time
    elif token1=="W1":
        calendar["f1_t"]=(-np.log(np.random.rand()) / lambda_fail)+ current_time
        
    return calendar

def sys(next_transition,token1,token2,token3,n_failure):
    if next_transition == "d_t":
        token2="S2"
        token3="D"
    elif next_transition=="n_ot_t":
        token2 = "W2"
        token3="N"
    elif next_transition =="n_t":
        token2 = "U2"
        token3="N"
    elif next_transition =="f1_t":
        token1="F1"
        n_failure+=1
    elif next_transition =="g1_t":
        token1="W1"
        token2="W2"
    elif next_transition=="s2_t":
        token2="S2"
    return token1,token2,token3,n_failure


def simulation(t):
    first_failure_time =0
    first_failure =False
    current_time =0
    n_failure=0
    token1 = "W1"
    token2 = "U2"
    token3 ="N"
    down_time =0
    failure_time =0
    calendar = calendar_transition(token1,token2,token3,current_time)
    
    while current_time <t :
        next_transition = min(calendar, key=calendar.get)
        t_next = calendar[next_transition]
        if t_next>=t:
            
            break
        
        if token1=="F1":
            failure_time+=current_time
            down_time+= t_next-current_time # time of repair - time of the last state that was total failure
        if token1=="F1" and first_failure==False:
            first_failure_time=current_time
            first_failure = True
        current_time = t_next #put the current time as the time for the next event
        token1,token2,token3,n_failure = sys(next_transition,token1,token2,token3,n_failure)
        calendar =calendar_transition(token1,token2,token3,current_time)


    
    return down_time,failure_time,n_failure,first_failure_time


lambda_fail = 0.04 #per hour
repair_time =1

n=8000


mission_times = np.linspace(0,200,40)
unavailability_values=[]
avail_values =[]
rel_val = []
for t in mission_times:
    count=0
    data =[]
    first_failure_time =[]
    n_failures = 0
    total_down_time=0
    total_failure_time=0
    for i in range(1,n):
        d_t,f_t,n_f,n_f_f =simulation(t)
        total_down_time+=d_t
        total_failure_time+=f_t
        n_failures+=n_f
        first_failure_time.append(n_f_f)
        if n_f_f ==0:
            count+=1
    rel_val.append(count/n)
    mttf = np.mean(first_failure_time)
    mnf = n_failures/n
    print("--------------------------------------------------------------------------------------")
    print(f"Mean number of failures : {mnf}")
    print(f"MTTF :{mttf} for t:{t}")
    unavailability_values.append(total_down_time/(n*t))
    avail_values.append(1-total_down_time/(n*t))


mttf1 = np.trapz(rel_val,mission_times)


print(f"yo : {mttf1}")
# plt.plot(mission_times, unavailability_values, label="Unavailability")
# plt.plot(mission_times, avail_values, label="Availability")
plt.plot(mission_times, rel_val, label="Reliability")
plt.xlabel("Mission Time")
plt.ylabel("reliability")
plt.title("System reliability Over Time")
plt.legend()
plt.grid()
plt.show()



