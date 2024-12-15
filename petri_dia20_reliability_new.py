import numpy as np
import matplotlib.pyplot as plt


def calendar_transition(token1,token2,current_time,fail_state1):
    # define calendar of events by doing each state possibilities and ensuring that the transition can be fired
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
    
    return calendar

def sys(next_transition,token1,token2,fail_state1):
    if next_transition == "f1_t":
        token1="F1"
        fail_state1=True
    elif next_transition =="g1_t":
        token1 = "W1"
        fail_state1=False
    elif next_transition =="s2_t":
        token2="W2"
    elif next_transition =="w2_t":
        token2="S2"
    elif next_transition =="f2_t":
        token2 = "F2"
    return token1,token2,fail_state1


def simulation(t,lambda_1,mu_1,lambda_2):
    current_time =0
    
    fail_state1 =False
    token1 = "W1"
    token2 = "S2"
    calendar = calendar_transition(token1,token2,current_time,fail_state1)
    while current_time <t :
        next_transition = min(calendar, key=calendar.get)
        t_next = calendar[next_transition]
        
        if t_next>=t:
            break
        current_time = t_next
        
        token1,token2,fail_state1 = sys(next_transition,token1,token2,fail_state1)
        if token1=="F1" and token2=="F2":
            
            return 1
        calendar =calendar_transition(token1,token2,current_time,fail_state1)

        
    return 0


failure_count = 0
repair_count = 0

lambda_1 = 0.1
mu_1 = 0.1
lambda_2 = 0.05

#t=100
n=10000
n_values = [1000,5000,10000,25000]

mission_times = np.linspace(0,300,40)

rel_data = []
for n in range(0,len(n_values)):
    rel_data.append([])
    for t in mission_times:
        count=0
        for i in range(1,n_values[n]):
            count+=simulation(t,lambda_1,mu_1,lambda_2)
        rel_data[n].append(1-count/n_values[n])
    mttf = np.trapz(rel_data[n],mission_times)


print(f"{mttf}")
for n in range(0,len(n_values)):
    plt.plot(mission_times, rel_data[n], label=f"n = {n_values[n]}")
plt.xlabel("Mission Time (hours)")
plt.ylabel("Reliability")
plt.title("System Reliability Over Time")
plt.legend()
plt.grid()
plt.show()

