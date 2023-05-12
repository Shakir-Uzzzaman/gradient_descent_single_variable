import numpy as np
def mean_squared_error(y_true,y_predict):
    cost=np.sum((y_true-y_predict)**2)/len(y_true)
    return cost
def gradient_descent(x,y,learnig_rate=0.0001,itt=1000,th=1e-6):
    current_weight=0.1
    current_bias=0.01
    itt=itt
    learning_rate=0.0001
    n=float(len(x))
    cost=[]
    weight=[]
    pr_cost=None
    for i in range(itt):
        y_prediction=(x*current_weight+current_bias)
        current_cost=mean_squared_error(y,y_prediction)
        if pr_cost and abs(pr_cost-current_cost)<=th:
            break
        pr_cost=current_cost
        cost.append(current_cost)
        weight.append(current_weight)
        weight_der=-(2/n) * sum(x*(y-y_prediction))
        bias_der=-(2/n) * sum(y-y_prediction)
        current_weight= current_weight-learning_rate*weight_der
        current_bias=current_bias-learning_rate*bias_der
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
		       {current_weight}, Bias {current_bias}")
    
    return current_weight,current_bias
def main():
    x=np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
           55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
           45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
           48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    y=np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
           78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
           55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
           60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
    est_weight,est_bias=gradient_descent(x, y,itt=2000)
    print(f"estimated weight:{est_weight}\nestimated bias:{est_bias}")
    
if __name__=="__main__":
    main()    

	

    
    

