import numpy as np
import matplotlib.pyplot as plt

# For ploting availability there is a need to modify the petri net
#need to add repair state, we will have to destination for the component 2 depending on whether or not component 1 is repaired
def calendar_transition(token1,token2,current_time,fail_state1):
    calendar ={}
    if token1=="W1":
        calendar["f1_t"]=(-np.log(np.random.rand()) / lambda_1)+ current_time
    elif token1=="F1":
        calendar["g1_t"]=(-np.log(np.random.rand()) / mu_1)+ current_time
    if token2=="S2" and fail_state1:
        calendar["s2_t"] = (current_time)
    elif token2=="W2" and not (fail_state1):
        calendar["w2_t"] = (current_time)
    elif token2=="W2" and fail_state1:
        calendar["f2_t"] = (-np.log(np.random.rand()) / lambda_2)+ current_time
    elif token2=="F2":
        calendar["g2_t"] = (-np.log(np.random.rand()) / mu_2)+ current_time
    elif token2=="Rep2":
        calendar["rw2_t"]=current_time

    return calendar

def sys(next_transition,token1,token2,fail_state1,n_failure_1,n_failure_2):
    if next_transition == "f1_t":
        token1="F1"
        fail_state1=True
        n_failure_1+=1
    elif next_transition =="g1_t":
        token1 = "W1"
        fail_state1=False
    elif next_transition =="s2_t":
        token2="W2"
    elif next_transition =="w2_t":
        token2="S2"
    elif next_transition =="f2_t":
        token2 = "F2"
        n_failure_2+=1
    elif next_transition=="g2_t":
        token2="Rep2"
    elif next_transition=="rw2_t" and fail_state1:
        token2="W2"
    elif next_transition=="rw2_t" and not fail_state1:
        token2="S2"
    return token1,token2,fail_state1,n_failure_1,n_failure_2


def simulation(t,lambda_1,mu_1,lambda_2):
    n_failure_tot=0
    first_failure_time=0
    first_failure=False
    n_failure_1=0
    n_failure_2=0
    current_time =0
    fail_state1 =False
    token1 = "W1"
    token2 = "S2"
    down_time =0
    failure_time =0
    calendar = calendar_transition(token1,token2,current_time,fail_state1)
    while current_time <t :
        next_transition = min(calendar, key=calendar.get)
        t_next = calendar[next_transition]
        if t_next>=t:
            break
        
        
        if token1=="F1" and token2=="F2":
            failure_time+=current_time
            n_failure_tot+=1
            down_time+= t_next-current_time # time of repair - time of the last state that was total failure
        if token1=="F1" and token2=="F2" and first_failure==False:
            
            first_failure_time=current_time
            first_failure = True
        token1,token2,fail_state1,n_failure_1,n_failure_2 = sys(next_transition,token1,token2,fail_state1,n_failure_1,n_failure_2)
        current_time = t_next #put the current time as the time for the next event
        calendar =calendar_transition(token1,token2,current_time,fail_state1)
    
    return down_time,failure_time,n_failure_1,n_failure_2,n_failure_tot,first_failure_time


failure_count = 0
repair_count = 0

lambda_1 = 0.1
mu_1 = 0.1
lambda_2 = 0.05
mu_2 = 0.05
#t=100
n=7000
n_values = [10000]

mission_times = np.linspace(0,400,400)
unavailability_values=[]
avail_data =[]
mttf_data=[]
mnr1_data=[]
for n in range(0,len(n_values)):
    # avail_data.append([])
    # unavailability_values.append([])
    for t in mission_times:
        data =[]
        first_failure_time =[]
        tot_n_failure_tot=0
        tot_n_failure_1=0
        tot_n_failure_2=0
        total_down_time=0
        total_failure_time=0
        for i in range(1,n_values[n]):
            d_t,f_t,n_f_1,n_f_2,n_f_tot,f_f_time =simulation(t,lambda_1,mu_1,lambda_2)
            total_down_time+=d_t
            total_failure_time+=f_t
            tot_n_failure_1+=n_f_1
            tot_n_failure_2+=n_f_2
            tot_n_failure_tot+=n_f_tot
            data.append(n_f_1)
            first_failure_time.append(f_f_time)

        mttf = np.mean(first_failure_time)
        mttf_data.append(mttf)
        mnr1 = tot_n_failure_1/n_values[n]
        mnr2 = tot_n_failure_2/n_values[n]
        mnrtot = tot_n_failure_tot/n_values[n]
        std_failures = np.std(data, ddof=1)

        # print("--------------------------------------------------------------------------------------")
        # print(f"Mean number of failures of component 1 : {mnr1}, component 2 : {mnr2} for t :{t}")
        # print(f"Mean number of total failure of the system : {mnrtot}, for t :{t}")
        # print(f"MTTF :{mttf} for t:{t}")
        # print(f"Standard Deviation: {std_failures:.2f}")
        # print(f"95% Confidence Interval: [{lower_ci:.2f}, {upper_ci:.2f}]")
        # unavailability_values[n].append(total_down_time/(n_values[n]*t))
        # avail_data[n].append(1-total_down_time/(n_values[n]*t))


# for n in range(0,len(n_values)):
#     plt.plot(mission_times, avail_data[n], label=f"Availability n = {n_values[n]}")
#     plt.plot(mission_times, unavailability_values[n], linestyle='--', label=f"Unavailability n = {n_values[n]}")


# plt.plot(mission_times, unavailability_values, label="Unavailability")
plt.plot(mission_times, mttf_data, label="MTTF")
plt.xlabel("Mission Time")
plt.ylabel("Mean Time To Failure")
plt.title("Mean Time to Failure Over Time")
plt.legend()
plt.grid()
plt.show()

